import argparse
import pandas as pd
import evaluator
import copy

from os import listdir
from os.path import isfile, join
from itertools import combinations

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--csv_folder', type=str, default='checkpoint/')
    parser.add_argument('--true', type=str, default='dataset/a415f-white/side_annotation.csv')
    parser.add_argument('--overkill', type=float, default=0.25)

   

    args = parser.parse_args()
    return args



class Ensemble_Method():
    def __init__(self, args):
        self.underkill_ratio = 1.0
        self.overkill_ratio = 1.0
        self.checked = False
        self.trues = pd.read_csv(args.true, sep=';')
        self.models = args.csv_folder
    
    def evaluate(self, origin_probs, thresh):
        probs = copy.deepcopy(origin_probs)
        
        prediction = probs.ng_prob >= thresh
        prediction = prediction.map(lambda b: 1 if b else 0)

        probs.insert(3, 'prediction', prediction)

        merged = pd.merge(self.trues, probs, on='patch_filename')
        sample_id = merged.patch_filename.map(lambda fname: fname.split('.')[2])
        merged.insert(0, 'sample_id', sample_id)
        
        summary = evaluator.get_summary(merged) 
        self.underkill_ratio = summary['stupid_fn_sample'][0] / summary['stupid_fn_sample'][1]
        self.overkill_ratio = summary['fp_sample'][0] / summary['fp_sample'][1]
        summary['threshold'] = thresh
        summary['model'] = "models in {}".format(self.models)
        return summary


    def evaluate_vote(self, origin_probs,preds): #maybe 
        probs = copy.deepcopy(origin_probs)
        probs.insert(3, 'prediction', preds)

        merged = pd.merge(self.trues, probs, on='patch_filename')
        sample_id = merged.patch_filename.map(lambda fname: fname.split('.')[2])
        merged.insert(0, 'sample_id', sample_id)

        summary = evaluator.get_summary(merged)
        self.underkill_ratio = summary['stupid_fn_sample'][0] / summary['stupid_fn_sample'][1]
        self.overkill_ratio = summary['fp_sample'][0] / summary['fp_sample'][1]
        return summary


def average_ensemble(tbe):
    averaged_probs = copy.deepcopy(tbe[0])
    
    ng_list= [probs.ng_prob for probs in tbe]
    ok_list= [probs.ok_prob for probs in tbe]
    
    averaged_probs.ng_prob = sum(ng_list)/len(ng_list)
    averaged_probs.ok_prob = sum(ok_list)/len(ok_list)
    
    return averaged_probs

def max_ensemble(tbe):
    max_probs = copy.deepcopy(tbe[0])
    
    ng_list= [probs.ng_prob for probs in tbe]

    for x in range(len(ng_list[0])):
        ng_sub = []
        for y in range(len(tbe)):
            ng_sub.append(ng_list[y][x])
        max_probs.ng_prob[x] = max(ng_sub)
    max_probs.ok_prob = 1 - max_probs.ng_prob

    return max_probs

def vote_ensemble(tbe, thresh): # #tobechecked FIXME
    preds_list = []
    for probs in tbe:
        prediction = probs.ng_prob >= thresh
        prediction = prediction.map(lambda b: 1 if b else 0)
        preds_list.append(prediction)

    
    voted_preds = copy.deepcopy(preds_list[0])
    for x in range(len(voted_preds)):
        plist = []
        for y in range(len(tbe)):
            plist.append(preds_list[y][x])
        num_ok = plist.count(0)
        num_ng = plist.count(1)
        if num_ok > num_ng:
            voted_preds[x] = 0
        else:
            voted_preds[x] = 1 
    return voted_preds


def make_combinations(n): 
    # if n = 3, [[0], [1], [2], [0,1], [0,2], [1,2], [0,1,2]]
    combinations_list = []
    for x in range(1,n+1):
        combinations_list += list(combinations(range(n),x))

    return combinations_list 
    

def ensemble(args):

    folder = args.csv_folder
    probs_list = []
    model_list = []
    for f in listdir(folder):
        if isfile(join(folder,f)):
            model_list.append(f)
            probs_list.append(pd.read_csv(join(folder,f), sep =';')) #[ probs[0], probs[1], ...]

    num_models = len(probs_list)

    comb_list = make_combinations(num_models)

    best_ensemble_info = {"underkill_ratio":1.0, "ensemble_method":None, "ensemble_indices":None}
    best_summary = None
    for comb in comb_list:
        print( "----------{} out of {} models are being ensembled...----------".format(len(comb),num_models))
        to_be_ensembled = [probs_list[idx] for idx in comb]
     
        # averaging
        averaged_probs = average_ensemble(to_be_ensembled)
        averaging = Ensemble_Method(args)

        # maximizing
        maximized_probs = max_ensemble(to_be_ensembled)
        maximizing = Ensemble_Method(args)

        # voting
        voting = Ensemble_Method(args) 

        methods_results_dict = {"averaging":1.0, "maximizing":1.0, "voting":1.0}
        methods_summary_dict = {"averaging": None, "maximizing":None, "voting":None}
        methods_count = 0
        threshold = 0.0
        while 1:
            averaging_summary = averaging.evaluate(averaged_probs,threshold)
            if averaging.overkill_ratio <= args.overkill and not averaging.checked:
                methods_results_dict["averaging"] = averaging.underkill_ratio
                methods_summary_dict["averaging"] = averaging_summary
                averaging.checked = True
                methods_count +=1

            maximizing_summary = maximizing.evaluate(maximized_probs,threshold)
            if maximizing.overkill_ratio <= args.overkill and not maximizing.checked:
                methods_results_dict["maximizing"] = maximizing.underkill_ratio
                methods_summary_dict["maximizing"] = maximizing_summary
                maximizing.checked = True
                methods_count +=1

            voted_preds = vote_ensemble(to_be_ensembled, threshold)
            voting_summary = voting.evaluate_vote(to_be_ensembled[0],voted_preds)# just put the first model for evaluation. It is meaningless.
            if voting.overkill_ratio <= args.overkill and not voting.checked:
                methods_results_dict["voting"] = voting.underkill_ratio
                methods_summary_dict["voting"] = voting_summary
                voting.checked = True
                methods_count +=1

            if methods_count == len(methods_results_dict):
                min_value = min(methods_results_dict.values())
                print("model", comb, "underkill ratio: ", min_value)
                if best_ensemble_info["underkill_ratio"] > min_value:  
                    best_ensemble_info["ensemble_method"] = min(methods_results_dict, key = methods_results_dict.get)
                    best_ensemble_info["underkill_ratio"] = min_value
                    best_ensemble_info["ensemble_indices"] = comb
                    best_summary = methods_summary_dict[best_ensemble_info["ensemble_method"]]
                    
                break
            threshold += 0.01
            
                

    return best_ensemble_info, best_summary, model_list


        





if __name__ == '__main__':
    args = get_args()
    
    info, summary, model_list = ensemble(args)
    print(info)
    for idx in info["ensemble_indices"]:
        print(model_list[idx])
    print("were ensembled")
    evaluator.print_summary(summary)

