import importlib
import lzma
import os
import pickle
import sys

import numpy as np
import PIL.Image
import torch


# %%
class Attributes:
    pass

class UnitTest:
    def __init__(self, 
                 easyocr_module, 
                 test_data = "./data/EasyOcrUnitTestPackage.pickle",
                 image_data_dir = "../examples", 
                 verbose = 0, 
                 numeric_acceptance_error = 0.1):
        
        self.verbose = verbose
      
        easy_ocr_init = os.path.join(easyocr_module, "__init__.py")
        if not os.path.isfile(easy_ocr_init):
            raise FileNotFoundError("Invalid easyocr_module. The directory should contain __init__.py.")
        
        spec = importlib.util.spec_from_file_location("easyocr", easy_ocr_init)
        easyocr = importlib.util.module_from_spec(spec)
        sys.modules["easyocr"] = easyocr
        spec.loader.exec_module(easyocr)
        
        self.easyocr = easyocr
        if not hasattr(self.easyocr, 'utils'):
            self.easyocr.utils = importlib.import_module('easyocr.utils')
        if not hasattr(self.easyocr, 'detection'):
            self.easyocr.detection = importlib.import_module('easyocr.detection')
        if not hasattr(self.easyocr, 'recognition'):
            self.easyocr.recognition = importlib.import_module('easyocr.recognition')
        
        self.easyocr_dir = os.path.dirname(easyocr.__file__)
        
        print(f"Unit test is set for EasyOCR at {os.path.abspath(self.easyocr_dir)}")
        
        self.image_data_dir = image_data_dir
        
        self.set_data(test_data)
        self.set_easyocr()
        self.numeric_acceptance_error = numeric_acceptance_error
    
    def set_data(self, test_data):
        
        self.inputs = Attributes()
        
        with lzma.open(test_data, 'rb') as fid:
            solution_book = pickle.load(fid)
        self.test_book = solution_book['tests']

        if any(file not in os.listdir(self.image_data_dir) for file in solution_book['inputs']['images']):
            raise FileNotFoundError("Cannot find {} in {}.").format(', '.join([file for file in solution_book['inputs']['images'] 
                                                                               if file not in os.listdir(self.image_data_dir)], self.image_data_dir))
        images = {os.path.splitext(file)[0]: {
                        key: np.asarray(PIL.Image.open(os.path.join(self.image_data_dir, file)).crop(crop_box))[:,:,::-1] for (key,crop_box) in page.items() 
                        } for (file,page) in solution_book['inputs']['images'].items()}

        english_mini_bgr, english_mini_gray = self.easyocr.utils.reformat_input(images['english']['mini'])
        english_small_bgr, english_small_gray = self.easyocr.utils.reformat_input(images['english']['small'])
        images['english'].update({'mini_bgr': english_mini_bgr,
                                  'mini_gray': english_mini_gray,
                                  'small_bgr': english_small_bgr,
                                  'small_gray': english_small_gray,
                                  })

        self.inputs.images = self.dict2attr(images)
        self.inputs.easyocr_config = self.dict2attr(solution_book['inputs']['easyocr_config'])
    
    def dict2attr(self, dict_):
        attr = Attributes()
        [setattr(attr, key, self.dict2attr(value)) if isinstance(value, dict) else setattr(attr, key, value) for (key,value) in dict_.items()]        
        return attr

    def count_parameters(self, model):
        return sum([param.numel() for param in model.parameters()])
    
    def get_weight_norm(self, model):
        with torch.no_grad():
            return sum([param.norm() for param in model.parameters()]).cpu().item()

    def get_nested_attr(self, parent, attr):
        if len(attr.split(".")) == 1:
            return getattr(parent, attr)
        else:
            attrs = attr.split(".")
            parent = getattr(parent, attrs[0])
            attr = ".".join(attrs[1:])
            attr = self.get_nested_attr(parent, attr)
            return attr
    
    def easyocr_read_as(self, image, language):
        if not isinstance(language, list):
            language = [language]
        reader =  self.easyocr.Reader(language)
        _, pred, confidence = reader.readtext(image)[0]
        reader = None
        torch.cuda.empty_cache()
        return pred, confidence
    
    def set_easyocr(self):
        ocr = self.easyocr.Reader([self.inputs.easyocr_config.main_language])
        self.easyocr.ocr = ocr
   
    
    def validate(self, test, solution, dtype):
        if dtype == str:
            return test == solution
        elif np.issubdtype(dtype, np.integer) or np.issubdtype(dtype, np.inexact):
            return abs(1-test/solution) < self.numeric_acceptance_error
        elif dtype == dict:
            return self.are_dicts_equal(test, solution)
        elif dtype == list or dtype == tuple:
            return self.are_lists_equal(test, solution)
        elif dtype == np.ndarray or dtype == torch.Tensor:
            return (abs(1-test/solution) < self.numeric_acceptance_error).all()
        else:
            raise TypeError(f"Unsupport data type ({dtype}) to validate. Supporting types are str, int, float, dict, list, np.ndarray, or torch.Tensor")
    
    def are_dicts_equal(self, test, solution):
        if test.keys() == solution.keys():
            return all(self.validate(test[key], solution[key], type(solution[key])) for key in solution)
        else:
            return False
    
    def are_lists_equal(self, test, solution):
        if len(test) == len(solution):
            return all(self.validate(tt, ss, type(ss)) for (tt,ss) in zip(test, solution))
        else:
            return False

    def is_list_or_tuple(self, test):
        return isinstance(test, (list, tuple))

    #Should check length of results/solutions/dtypes 
    def validate_all(self, results, solutions, dtypes):
        if not isinstance(results, list):
            results = [results]
        if not isinstance(solutions, list):
            solutions = [solutions]
        if not isinstance(dtypes, list):
            dtypes = [dtypes]
        
        
        validation = []
        for (result, solution, dtype) in zip(results, solutions, dtypes):
            if (not self.is_list_or_tuple(result)
                and not self.is_list_or_tuple(result)
                and not self.is_list_or_tuple(result)
                ): 
                validation.append(self.validate(result, solution, type(solution)))
            elif(self.is_list_or_tuple(result)
                and self.is_list_or_tuple(result)
                and self.is_list_or_tuple(result)
                ):
                validation.append(self.validate_all(results, solutions, type(solution)))
            else:
                raise Exception("No exception to re-raise")
        return all(validation)

    def do_test(self, verbose = None):
        if verbose is not None:
            self.verbose = verbose
        
        num_module_to_test = len(self.test_book)
        num_module_pass = 0
        print(f"Testing EasyOCR: {num_module_to_test:d} modules will be tested.\n")
        for name,tests in self.test_book.items():
            num_test = len(tests)
            num_passed = 0
            min_pass = sum([test['severity'] == 'Error' for test in tests.values()])
            if self.verbose > 0:
                print(f"##Testing module {name}: {num_test:d} tests will be performed.")
            for test_id, test in tests.items():
                if self.verbose > 1:
                    print("#### {}: {}".format(test_id, test['description']))
                
                if test['method'].startswith('unit_test.'):
                    test['method'] = '.'.join(test['method'].split('.')[1:])
                test_method = self.get_nested_attr(self, test['method'])
                
                test['input'] = [(self.get_nested_attr(self, '.'.join(input_.split('.')[1:]))
                                 if input_.startswith('unit_test.') else input_) if isinstance(input_, str) else input_ for input_ in test['input']]
                # reference inputs were recorded on a CUDA machine; run on the reader's device instead
                active_device = getattr(getattr(self.easyocr, 'ocr', None), 'device', 'cpu')
                test['input'] = [active_device if isinstance(input_, str) and input_ == 'cuda' else input_ for input_ in test['input']]
                if verbose > 3:
                    print("###### Input: {}".format(test['input']))
                results = test_method(*test['input'])
                if verbose > 2:
                    print("###### Expected output: {}".format(test['output']))
                    print(f"###### Received output: {results}")
                test_result = self.validate(results, test['output'], type(test['output']))
                if test_result:
                    num_passed += 1
                    if self.verbose > 1:
                        print(f"#### Passed. [{num_passed:d}/{num_test:d}]")
                else:
                    if test['severity'] == "Warning": 
                        num_passed += 1
                        if self.verbose > 1:
                            print(f"#### Passed. [{num_passed:d}/{num_test:d}]")
                        if self.verbose > 2:
                            print("##### Warning: While the result is considered as passed, the test yields results ({}) \
                              that are different from the expected values ({}). It is strongly recommended to make sure \
                              that this is expected.".format(results, test['output']))
                    else:
                        if self.verbose > 1:
                            print("#### Failed")
                        if self.verbose > 2:
                            print("##### The test yields results ({}) which are different from the expected values ({}).")
        
            if num_passed >= min_pass:
                num_module_pass += 1
                if self.verbose > 0: 
                    print(f"##Module {name}: Passed.\n")
            else:
                print(f"##Module {name}: Failed.\n")
        
        print("#"*50)
        if num_module_pass >= num_module_to_test:
            print("Testing completed:\n Final result: Passed.")
        else:
            print("Testing completed:\n Final result: Failed.")
        



























        