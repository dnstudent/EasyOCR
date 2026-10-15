import argparse

import numpy as np
import torch

import easyocr

# onnxruntime/CoreML sessions can segfault when destroyed during interpreter
# shutdown; holding them until process exit sidesteps the teardown entirely
_ORT_SESSIONS = []


class _RecognizerForExport(torch.nn.Module):
    """The CTC recognizer takes a dummy ``text`` argument at inference time;
    drop it so the exported ONNX graph has a single image input."""

    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, image):
        return self.model(image, None)


def _ort_session(path, prefer_coreml=True):
    import onnxruntime as ort

    providers = ["CoreMLExecutionProvider", "CPUExecutionProvider"] if prefer_coreml else ["CPUExecutionProvider"]
    providers = [p for p in providers if p in ort.get_available_providers()]
    session = ort.InferenceSession(path, providers=providers)
    _ORT_SESSIONS.append(session)
    return session


def _validate_onnx(path, torch_outputs, ort_inputs):
    """Compare ONNX Runtime output against PyTorch.

    The CPU provider is checked with a tight tolerance (it must match fp32
    torch closely); when the CoreML provider is available it is also run with
    a looser tolerance, since CoreML may compute in float16.
    """
    cpu_session = _ort_session(path, prefer_coreml=False)
    ort_outputs = cpu_session.run(None, ort_inputs)
    for torch_out, ort_out in zip(torch_outputs, ort_outputs):
        np.testing.assert_allclose(torch_out, ort_out, rtol=1e-3, atol=1e-4)

    try:
        coreml_session = _ort_session(path, prefer_coreml=True)
        if "CoreMLExecutionProvider" in coreml_session.get_providers():
            coreml_outputs = coreml_session.run(None, ort_inputs)
            max_diff = max(float(np.abs(np.asarray(t) - np.asarray(o)).max())
                           for t, o in zip(torch_outputs, coreml_outputs))
            # fp16 compute on Apple hardware: small absolute error on near-zero
            # logits, far below the 0.2-0.7 thresholds the detector applies later
            for torch_out, coreml_out in zip(torch_outputs, coreml_outputs):
                np.testing.assert_allclose(torch_out, coreml_out, rtol=5e-2, atol=2e-2)
            print(f"CoreMLExecutionProvider validated (max abs diff vs torch: {max_diff:.2e})")
    except (RuntimeError, ValueError) as e:
        # CoreML cannot compile every graph, e.g. the dynamic-width LSTM the
        # recognizer exports to; CPU validation above is the correctness gate
        print(f"note: CoreMLExecutionProvider could not run this graph ({type(e).__name__}); "
              f"CPU-provider validation above is authoritative")
    print("ONNXRuntime validation passed (CPU provider, rtol=1e-3)")


def export_detector(detector_onnx_save_path,
                    in_shape=None,
                    lang_list=None,
                    model_storage_directory=None,
                    user_network_directory=None,
                    download_enabled=True,
                    dynamic=True,
                    device="cpu",
                    quantize=False,
                    detector=True,
                    recognizer=True):
    if lang_list is None:
        lang_list = ["en"]
    if in_shape is None:
        in_shape = [1, 3, 608, 800]
    if dynamic is False:
        print('WARNING: it is recommended to use -d dynamic flag when exporting onnx')
    ocr_reader = easyocr.Reader(lang_list,
                                gpu=device != "cpu",
                                detector=detector,
                                recognizer=detector,
                                quantize=quantize,
                                model_storage_directory=model_storage_directory,
                                user_network_directory=user_network_directory,
                                download_enabled=download_enabled)

    dummy_input = torch.rand(in_shape).to(device)

    # forward pass
    with torch.no_grad():
        y_torch_out, feature_torch_out = ocr_reader.detector(dummy_input)
        dynamic_shapes = {"x": {0: "batch", 2: "height", 3: "width"}} if dynamic else None
        torch.onnx.export(ocr_reader.detector,
                          (dummy_input,),
                          detector_onnx_save_path,
                          input_names=['input'],
                          output_names=['output', 'feature'],
                          dynamic_shapes=dynamic_shapes,
                          verbose=False)

    # verify exported onnx model
    import onnx

    detector_onnx = onnx.load(detector_onnx_save_path)
    onnx.checker.check_model(detector_onnx)
    print(f"Model Inputs:\n {detector_onnx.graph.input}\n{'*'*80}")
    print(f"Model Outputs:\n {detector_onnx.graph.output}\n{'*'*80}")

    # onnx inference validation
    ort_session = _ort_session(detector_onnx_save_path)

    def to_numpy(tensor):
        return tensor.detach().cpu().numpy()

    ort_inputs = {ort_session.get_inputs()[0].name: to_numpy(dummy_input)}

    print(f"torch outputs: y_torch_out.shape={tuple(y_torch_out.shape)} feature_torch_out.shape={tuple(feature_torch_out.shape)}")
    _validate_onnx(detector_onnx_save_path,
                   [to_numpy(y_torch_out), to_numpy(feature_torch_out)],
                   ort_inputs)

    print(f"Model exported to {detector_onnx_save_path} and tested with ONNXRuntime, and the result looks good!")


def export_recognizer(recognizer_onnx_save_path,
                      lang_list=None,
                      recog_network='standard',
                      model_storage_directory=None,
                      user_network_directory=None,
                      download_enabled=True,
                      dynamic=True,
                      device="cpu",
                      quantize=False,
                      in_width=600):
    """Export the CTC recognizer to ONNX.

    The exported graph takes (N, 1, imgH, in_width) float inputs: the batch
    dimension is dynamic, the width is fixed, so crops must be right-padded
    to ``in_width`` first (the torch inference path pads crops the same way).
    A truly dynamic width is not expressible: the LSTM flattening bakes the
    sequence length into the graph, and CoreML only compiles the LSTM
    subgraph at a fixed sequence length.
    """
    if lang_list is None:
        lang_list = ["en"]
    ocr_reader = easyocr.Reader(lang_list,
                                gpu=device != "cpu",
                                detector=False,
                                recognizer=True,
                                quantize=quantize,
                                recog_network=recog_network,
                                model_storage_directory=model_storage_directory,
                                user_network_directory=user_network_directory,
                                download_enabled=download_enabled)

    # Attention-based (generation1) recognizers decode with the text input at
    # every step; only the CTC models can run as a plain image -> logits graph.
    if ocr_reader.recognizer.__class__.__name__ != 'Model':
        raise ValueError('ONNX export supports the CTC recognizer (generation2 / custom CTC) only')

    from easyocr.config import imgH
    model = _RecognizerForExport(ocr_reader.recognizer).to(device).eval()
    dummy_input = torch.rand(1, 1, imgH, in_width).to(device)

    with torch.no_grad():
        torch_out = model(dummy_input)
        dynamic_shapes = {"image": {0: "batch"}} if dynamic else None
        torch.onnx.export(model,
                          (dummy_input,),
                          recognizer_onnx_save_path,
                          input_names=['image'],
                          output_names=['logits'],
                          dynamic_shapes=dynamic_shapes,
                          verbose=False)

    import onnx

    recognizer_onnx = onnx.load(recognizer_onnx_save_path)
    onnx.checker.check_model(recognizer_onnx)

    # CoreML cannot compile the LSTM subgraph, so validate on the CPU provider
    ort_session = _ort_session(recognizer_onnx_save_path, prefer_coreml=False)
    ort_inputs = {'image': dummy_input.cpu().numpy()}
    _validate_onnx(recognizer_onnx_save_path, [torch_out.cpu().numpy()], ort_inputs)
    ort_out = ort_session.run(None, ort_inputs)[0]

    # dynamic batch check: a larger batch must run and keep the time axis
    batch2 = torch.rand(3, 1, imgH, in_width).to(device)
    with torch.no_grad():
        batch2_torch = model(batch2).cpu().numpy()
    batch2_ort = ort_session.run(None, {'image': batch2.cpu().numpy()})[0]
    np.testing.assert_allclose(batch2_torch, batch2_ort, rtol=1e-3, atol=1e-4)
    assert batch2_ort.shape[:1] == (3,) and batch2_ort.shape[1:] == ort_out.shape[1:], \
        'dynamic batch not honoured'

    print(f"Recognizer exported to {recognizer_onnx_save_path} "
          f"(input 1x{imgH}x{in_width}, dynamic batch verified, width fixed: pad crops to {in_width})")


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('-l', '--lang_list',
                        nargs='+', type=str,
                        default=["en"],
                        help='-l en ch_sim ... (language lists for easyocr)')
    parser.add_argument('-s', '--detector_onnx_save_path', type=str,
                        default="detector_craft.onnx",
                        help="export detector onnx file path ending in .onnx. "
                        "Pass 'None' to avoid exporting detector")
    parser.add_argument('-r', '--recognizer_onnx_save_path', type=str,
                        default=None,
                        help="export CTC recognizer onnx file path ending in .onnx")
    parser.add_argument('-d', '--dynamic',
                        action='store_true',
                        help="Dynamic input output shapes")
    parser.add_argument('-is', '--in_shape',
                        nargs='+', type=int,
                        default=[1, 3, 608, 800],
                        help='-is 1 3 608 800 (bsize, channel, height, width)')
    parser.add_argument('-m', '--model_storage_directory', type=str,
                        help="model storage directory for craft model")
    parser.add_argument('-u', '--user_network_directory', type=str,
                        help="user model storage directory")
    args = parser.parse_args()
    if args.detector_onnx_save_path == "None":
        args.detector_onnx_save_path = None
    if len(args.in_shape) != 4:
        raise ValueError(
            "Input shape must have four values (bsize, channel, height, width) eg. 1 3 608 800")
    return args


def main():
    args = parse_args()
    if args.detector_onnx_save_path:
        export_detector(detector_onnx_save_path=args.detector_onnx_save_path,
                        in_shape=args.in_shape,
                        lang_list=args.lang_list,
                        model_storage_directory=args.model_storage_directory,
                        user_network_directory=args.user_network_directory,
                        dynamic=args.dynamic)
    if args.recognizer_onnx_save_path:
        export_recognizer(recognizer_onnx_save_path=args.recognizer_onnx_save_path,
                          lang_list=args.lang_list,
                          model_storage_directory=args.model_storage_directory,
                          user_network_directory=args.user_network_directory,
                          dynamic=args.dynamic)


if __name__ == "__main__":
    main()
