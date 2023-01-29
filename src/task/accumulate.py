import os
import json

DEFAULT_FOLDER = '/home/azureuser/cloudfiles/code/Users/xiaoyuz/oneshot_flaml_deeplearning/default'
RESULT_FOLDER = f'{DEFAULT_FOLDER}/result_dnn'
CONFIG_FOLDER = f'{DEFAULT_FOLDER}/result'

OUTPUT_FOLDER = f'{DEFAULT_FOLDER}/dnn'

# generate results.csv
result_path = f'{OUTPUT_FOLDER}/results.csv'
result_files = filter(lambda x: x.endswith('json'), os.listdir(RESULT_FOLDER))

with open(result_path, 'w') as fs:
    fs.write('task,type,result,fold,params\r\n')
    for result in result_files:
        result = f'{RESULT_FOLDER}/{result}'
        with open(result, 'r') as rs:
            result_obj = json.load(rs)
            for key in result_obj:
                obj = result_obj[key]
                modelJson = {'_modeljson': f'{RESULT_FOLDER}/{key}'}
                fs.write(f"{obj['task']},{obj['type']},{obj['result']},0,{modelJson}\r\n")

# format metafeatures.json
with open(f'{DEFAULT_FOLDER}/metafeatures.json', 'r') as f:
    metafeatures = json.load(f)
    with open(f'{DEFAULT_FOLDER}/metafeatures.csv', 'w') as fp:
        fp.write('Dataset,NumberOfInstances,NumberOfFeatures,NumberOfClasses\r\n')

        for dataset in metafeatures:
            obj = metafeatures[dataset]
            fp.write(f"{dataset},{obj['number_of_instances']},{obj['number_of_features']},{obj['number_of_classes']}\r\n")

