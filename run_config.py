import importlib
from pathlib import Path
import sys
import time
import os
import glob
import traceback
import socket
import numpy as np
import pandas as pd
from datetime import datetime
import yaml

import evaluation.loader as dl
from builtins import Exception
import pickle
import dill
import random
import gc

NOTIFY = False


def main(conf, out=None):
    '''
    Execute experiments for the given configuration path
        --------
        conf: string
            Configuration path. Can be a folder.
        out: string
            Output folder path for endless run listening for new configurations. 
    '''
    print('Checking {}'.format(conf))
    # updater.dispatcher.add_handler( CommandHandler('status', status) )

    file = Path(conf)
    if file.is_dir():
        if out is not None:
            ensure_dir(out + '/out.txt')

            send_message('waiting for configuration files in ' + conf)

            while True:
                print( 'waiting for configuration files in ', conf )

                list = glob.glob(conf + '/' + '*.yml')
                for conf_file in list[:1]:
                    try:
                        file = Path(conf_file)
                        print('processing config', conf_file)
                        send_message('processing config ' + conf_file)
                        stream = open(str(file))
                        c = yaml.load(stream)
                        stream.close()

                        ts = time.time()
                        run_file(c)

                        print('finished config', conf_file, "for ", time.time() - ts, "secs.")
                        send_message('finished config ' + conf_file)
                        os.rename(conf_file, f'{out}/{time.strftime("%y%m%d_%H%M%S")}_O_{file.name}')

                    except (KeyboardInterrupt, SystemExit):

                        send_message('manually aborted config ' + conf_file)
                        os.rename(conf_file, f'{out}/{time.strftime("%y%m%d_%H%M%S")}_C_{file.name}')

                        raise

                    except Exception:
                        print('error for config ', conf_file)
                        os.rename(conf_file, f'{out}/{time.strftime("%y%m%d_%H%M%S")}_X_{file.name}')
                        send_exception('error for config ' + conf_file)
                        traceback.print_exc()
                
                time.sleep(60)

        else:
            print('processing folder ', conf)

            list = glob.glob(conf + '/' + '*.yml')
            for conf in list:
                try:

                    print('processing config', conf)
                    send_message('processing config ' + conf)

                    stream = open(str(Path(conf)))
                    c = yaml.load(stream)
                    stream.close()

                    run_file(c)

                    print('finished config', conf)
                    send_message('finished config ' + conf)

                except (KeyboardInterrupt, SystemExit):
                    send_message('manually aborted config ' + conf)
                    raise

                except Exception:
                    print('error for config ', conf)
                    send_exception('error for config' + conf)
                    traceback.print_exc()

            exit()


def run_file(conf):
    '''
    Execute experiments for one single configuration file
        --------
        conf: dict
            Configuration dictionary
    '''
    if conf['type'] == 'single':
        run_single(conf)
    elif conf['type'] == 'single_all':
        run_single_all(conf)
    elif conf['type'] == 'single_opt':
        run_single_opt(conf)
    elif conf['type'] == 'window':
        run_window(conf)
    elif conf['type'] == 'window_all':
        run_window_all(conf)
    elif conf['type'] == 'window_opt':
        run_window_opt(conf)
    elif conf['type'] == 'opt':
        run_opt(conf)
    else:
        print(conf['type'] + ' not supported')


def run_single_all(conf, slice=None):
    '''
    Evaluate the algorithms for a single split
        --------
        conf: dict
            Configuration dictionary
        slice: int
            Optional index for the window slice
    '''
    print('run test single')

    algorithms = create_algorithms_dict(conf['algorithms'])
    metrics_next = create_metric_list(conf['metrics_next'])
    metrics_multiple = create_metric_list(conf['metrics_multiple'])
    evaluation_next = load_evaluation('evaluation')
    evaluation_multiple = load_evaluation('evaluation_multiple')

    if 'opts' in conf['data']:
        train, test = dl.load_data_session(conf['data']['folder'], conf['data']['prefix'], slice_num=slice, **conf['data']['opts'])
    else:
        train, test = dl.load_data_session(conf['data']['folder'], conf['data']['prefix'], slice_num=slice)

    buys = pd.DataFrame()
    if 'buys' in conf['data'] and 'file_buys' in conf['data']:
        buys = dl.load_buys(conf['data']['folder'], conf['data']['file_buys'])  # load buy actions in addition

    for m in metrics_next:
        m.init(train)
        if hasattr(m, 'set_buys'):
            m.set_buys(buys, test)
    for m in metrics_multiple:
        m.init(train)
        if hasattr(m, 'set_buys'):
            m.set_buys(buys, test)
    results = {}


    count = 1
    for k, a in algorithms.items():
        print(f'{count} / {len(algorithms)}')
        count+=1
        ts = time.time()
        eval_algorithm_all(train, test, k, a, evaluation_next, evaluation_multiple, metrics_next, metrics_multiple, results, conf, slice=slice, iteration=slice)
        print(f"Time for Train + Evaluation: {time.time() - ts} secs.")


    print_results(results)
    write_results_csv(results, conf, iteration=slice)

    return results


def run_single(conf, slice=None):
    '''
    Evaluate the algorithms for a single split
        --------
        conf: dict
            Configuration dictionary
        slice: int
            Optional index for the window slice
    '''
    print('run test single')

    algorithms = create_algorithms_dict(conf['algorithms'])
    metrics = create_metric_list(conf['metrics'])
    evaluation = load_evaluation(conf['evaluation'])

    if 'opts' in conf['data']:
        train, test = dl.load_data_session(conf['data']['folder'], conf['data']['prefix'], slice_num=slice, **conf['data']['opts'])
    else:
        train, test = dl.load_data_session(conf['data']['folder'], conf['data']['prefix'], slice_num=slice)

    buys = pd.DataFrame()
    if 'buys' in conf['data'] and 'file_buys' in conf['data']:
        buys = dl.load_buys(conf['data']['folder'], conf['data']['file_buys'])  # load buy actions in addition

    for m in metrics:
        m.init(train)
        if hasattr(m, 'set_buys'):
            m.set_buys(buys, test)

    results = {}

    count = 1
    for k, a in algorithms.items():
        print(f'{count} / {len(algorithms)}')
        count+=1
        ts = time.time()
        eval_algorithm(train, test, k, a, evaluation, metrics, results, conf, slice=slice, iteration=slice)
        print(f"Time for Train + Evaluation: {time.time() - ts} secs.")

    print_results(results)
    write_results_csv(results, conf, iteration=slice)

    del algorithms
    del metrics
    del evaluation
    gc.collect()

    return results

def run_single_opt(conf, slice=None):
    '''
    Evaluate the algorithms for a single split
        --------
        conf: dict
            Configuration dictionary
        slice: int
            Optional index for the window slice
    '''
    print('run test single_opt')

    algorithms = create_algorithms_dict(conf['algorithms'])
    metrics = create_metric_list(conf['metrics'])
    evaluation = load_evaluation(conf['evaluation'])

    train_eval = True
    if 'opts' in conf['data']:
        train, test = dl.load_data_session(conf['data']['folder'], conf['data']['prefix'], train_eval=train_eval, slice_num=slice, **conf['data']['opts'])
    else:
        train, test = dl.load_data_session(conf['data']['folder'], conf['data']['prefix'], train_eval=train_eval, slice_num=slice)

    buys = pd.DataFrame()
    if 'buys' in conf['data'] and 'file_buys' in conf['data']:
        buys = dl.load_buys(conf['data']['folder'], conf['data']['file_buys'])  # load buy actions in addition

    for m in metrics:
        m.init(train)
        if hasattr(m, 'set_buys'):
            m.set_buys(buys, test)

    results = {}

    count = 1
    for k, a in algorithms.items():
        print(f'{count} / {len(algorithms)}')
        count+=1
        ts = time.time()
        eval_algorithm(train, test, k, a, evaluation, metrics, results, conf, slice=slice, iteration=slice)
        print(f"Time for Train + Evaluation: {time.time() - ts} secs.")

    print_results(results)
    write_results_csv(results, conf, iteration=slice)

    del algorithms
    del metrics
    del evaluation
    gc.collect()

    return results


def run_opt_single(conf, iteration, globals):
    '''
    Evaluate the algorithms for a single split
        --------
        conf: dict
            Configuration dictionary
        slice: int
            Optional index for the window slice
    '''
    print('run test opt single')

    algorithms = create_algorithms_dict(conf['algorithms'])
    for k, a in algorithms.items():
        aclass = type(a)
        if not aclass in globals:
            globals[aclass] = {'key': '', 'best': -1}

    metrics = create_metric_list(conf['metrics'])
    metric_opt = create_metric(conf['optimize'])
    metrics = metric_opt + metrics
    evaluation = load_evaluation(conf['evaluation'])

    train_eval = True
    if 'train_eval' in conf['data']:
        train_eval = conf['data']['train_eval']

    if 'opts' in conf['data']:
        train, test = dl.load_data_session(conf['data']['folder'], conf['data']['prefix'], train_eval=train_eval, **conf['data']['opts'])
    else:
        train, test = dl.load_data_session(conf['data']['folder'], conf['data']['prefix'], train_eval=train_eval)

    for m in metrics:
        m.init(train)

    results = {}

    for k, a in algorithms.items():
        ts = time.time()
        eval_algorithm(train, test, k, a, evaluation, metrics, results, conf, iteration=iteration, out=False)
        print(f"Time for Train + Evaluation: {time.time() - ts} secs.")

    write_results_csv(results, conf, iteration=iteration)

    for k, a in algorithms.items():
        aclass = type(a)
        current_value = results[k][0][1]
        if globals[aclass]['best'] < current_value:
            print('found new best configuration')
            print(k)
            print('improvement from {} to {}'.format(globals[aclass]['best'], current_value))
            send_message('improvement for {} from {} to {} in test {}'.format(k, globals[aclass]['best'], current_value, iteration))
            globals[aclass]['best'] = current_value
            globals[aclass]['key'] = k

    globals['results'].append(results)

    del algorithms
    del metrics
    del evaluation
    del results
    gc.collect()


def run_window(conf):
    '''
     Evaluate the algorithms for all slices
         --------
         conf: dict
             Configuration dictionary
     '''

    print('run test window')

    slices = conf['data']['slices']
    slices = list(range(slices))
    if 'skip' in conf['data']:
        for i in conf['data']['skip']:
            slices.remove(i)

    # For Each fold
    for i in slices:
        print('start run for slice ', str(i))
        send_message('start run for slice ' + str(i))
        results = run_single(conf, slice=i)
        if i == 0:
            # {'ar': [('Pre@20: ', 0.049), ('Rec@20: ', 0.398)], ... }
            avg_results = results
        else:
            # avg_results += results
            for k, l in results.items():
                for i in range(len(l)):
                    avg_results[k][i] = [avg_results[k][i][0], avg_results[k][i][1] + results[k][i][1]]
    # avg_results /= n-folds
    for k, l in avg_results.items():
        for i in range(len(l)):
            avg_results[k][i][1] /= len(slices)
    write_results_csv(avg_results, conf, extra='avg')

def run_window_all(conf):
    '''
     Evaluate the algorithms for all slices
         --------
         conf: dict
             Configuration dictionary
     '''

    print('run test window')

    slices = conf['data']['slices']
    slices = list(range(slices))
    if 'skip' in conf['data']:
        for i in conf['data']['skip']:
            slices.remove(i)

    # For Each fold
    for i in slices:
        print('start run for slice ', str(i))
        send_message('start run for slice ' + str(i))
        results = run_single_all(conf, slice=i)
        if i == 0:
            # {'ar': [('Pre@20: ', 0.049), ('Rec@20: ', 0.398)], ... }
            avg_results = results
        else:
            # avg_results += results
            for k, l in results.items():
                for i in range(len(l)):
                    avg_results[k][i] = [avg_results[k][i][0], avg_results[k][i][1] + results[k][i][1]]
    # avg_results /= n-folds
    for k, l in avg_results.items():
        for i in range(len(l)):
            avg_results[k][i][1] /= len(slices)
    write_results_csv(avg_results, conf, extra='avg')

def run_window_opt(conf):
    '''
     Evaluate the algorithms for first slices
         --------
         conf: dict
             Configuration dictionary
     '''

    print('run test window')

    slices = conf['data']['slices']
    slices = list(range(slices))
    if 'skip' in conf['data']:
        for i in conf['data']['skip']:
            slices.remove(i)

    # For Each fold
    for i in slices:
        print('start run for slice ', str(i))
        send_message('start run for slice ' + str(i))
        results = run_single_opt(conf, slice=i)
        if i == 0:
            # {'ar': [('Pre@20: ', 0.049), ('Rec@20: ', 0.398)], ... }
            avg_results = results
        else:
            # avg_results += results
            for k, l in results.items():
                for i in range(len(l)):
                    avg_results[k][i] = [avg_results[k][i][0], avg_results[k][i][1] + results[k][i][1]]
    # avg_results /= n-folds
    for k, l in avg_results.items():
        for i in range(len(l)):
            avg_results[k][i][1] /= len(slices)
    write_results_csv(avg_results, conf, extra='avg')


def run_opt(conf):
    '''
     Perform an optmization for the algorithms
         --------
         conf: dict
             Configuration dictionary
     '''

    iterations = conf['optimize']['iterations'] if 'optimize' in conf and 'iterations' in conf['optimize'] else 100
    start = conf['optimize']['iterations_skip'] if 'optimize' in conf and 'iterations_skip' in conf['optimize'] else 0
    print('run opt with {} iterations starting at {}'.format(iterations, start))

    globals = {}
    globals['results'] = []

    for i in range(start, iterations):
        print('start random test ', str(i))
        run_opt_single(conf, i, globals)

    global_results = {}
    for results in globals['results']:
        for key, value in results.items():
            global_results[key] = value

    write_results_csv(global_results, conf)


def eval_algorithm_all(train, test, key, algorithm, eval_next, eval_multiple, metrics_next, metrics_multiple, results, conf, slice=None, iteration=None, out=True):
    '''
    Evaluate one single algorithm
        --------
        train : Dataframe
            Training data
        test: Dataframe
            Test set
        key: string
            The automatically created key string for the algorithm
        algorithm: algorithm object
            Just the algorithm object, e.g., ContextKNN
        eval: module
            The module for evaluation, e.g., evaluation.evaluation_last
        metrics: list of Metric
            Optional string to add to the file name
        results: dict
            Result dictionary
        conf: dict
            Configuration dictionary
        slice: int
            Optional index for the window slice
    '''
    ts = time.time()
    print('----------------------------------------fit ', key)
    #send_message( 'training algorithm ' + key )

    if hasattr(algorithm, 'init'):
        algorithm.init(train, test, slice=slice)

    for m in metrics_next:
        if hasattr(m, 'start'):
            m.start(algorithm)
    for m in metrics_multiple:
        if hasattr(m, 'start'):
            m.start(algorithm)

    algorithm.fit(train, test)
    print(key, ' time: ', (time.time() - ts))

    if 'results' in conf and 'pickle_models' in conf['results']:
        try:
            save_model(key, algorithm, conf)
        except Exception:
            print('could not save model for ' + key)

    for m in metrics_next:
        if hasattr(m, 'start'):
            m.stop(algorithm)
    for m in metrics_multiple:
        if hasattr(m, 'start'):
            m.stop(algorithm)

    results[key] = []
    results[key] += eval_next.evaluate_sessions(algorithm, metrics_next, test, train)
    results[key] += eval_multiple.evaluate_sessions(algorithm, metrics_multiple, test, train)
    if out:
        write_results_csv({key: results[key]}, conf, extra=key, iteration=iteration)
    print_results({key: results[key]})

    #send_message( 'algorithm ' + key + ' finished ' + ( 'for slice ' + str(slice) if slice is not None else '' ) )

    algorithm.clear()


def eval_algorithm(train, test, key, algorithm, eval, metrics, results, conf, slice=None, iteration=None, out=True):
    '''
    Evaluate one single algorithm
        --------
        train : Dataframe
            Training data
        test: Dataframe
            Test set
        key: string
            The automatically created key string for the algorithm
        algorithm: algorithm object
            Just the algorithm object, e.g., ContextKNN
        eval: module
            The module for evaluation, e.g., evaluation.evaluation_last
        metrics: list of Metric
            Optional string to add to the file name
        results: dict
            Result dictionary
        conf: dict
            Configuration dictionary
        slice: int
            Optional index for the window slice
    '''
    ts = time.time()
    print('----------------------------------------fit ', key)
    #send_message( 'training algorithm ' + key )

    if hasattr(algorithm, 'init'):
        algorithm.init(train, test, slice=slice)

    for m in metrics:
        if hasattr(m, 'start'):
            m.start(algorithm)

    algorithm.fit(train, test)
    print(key, ' time: ', (time.time() - ts))

    if 'results' in conf and 'pickle_models' in conf['results']:
        try:
            save_model(key, algorithm, conf)
        except Exception:
            print('could not save model for ' + key)

    for m in metrics:
        if hasattr(m, 'start'):
            m.stop(algorithm)

    results[key] = eval.evaluate_sessions(algorithm, metrics, test, train)
    if out:
        write_results_csv({key: results[key]}, conf, extra=key, iteration=iteration)
    print_results({key: results[key]})

    #send_message( 'algorithm ' + key + ' finished ' + ( 'for slice ' + str(slice) if slice is not None else '' ) )

    algorithm.clear()


def write_results_csv(results, conf, iteration=None, extra=None):
    '''
    Write the result array to a csv file, if a result folder is defined in the configuration
        --------
        results : dict
            Dictionary of all results res[algorithm_key][metric_key]
        iteration; int
            Optional for the window mode
        extra: string
            Optional string to add to the file name
    '''

    if 'results' in conf and 'folder' in conf['results']:

        export_csv = conf['results']['folder'] + str(time.strftime("%y%m%d_%H%M%S")) + '_' + conf['data']['name']
        if extra is not None:
            export_csv += '.' + str(extra)
        if iteration is not None:
            export_csv += '.' + str(iteration)
        export_csv += '.csv'

        ensure_dir(export_csv)

        file = open(export_csv, 'w+')
        file.write('Metrics,')

        for k, l in results.items():
            for e in l:
                file.write(e[0])
                file.write(',')
            break

        file.write('\n')

        for k, l in results.items():
            file.write(k)
            file.write(',')
            for e in l:
                file.write(f'{e[1]:.4f}')
                file.write(',')
                if len(e) > 2:
                    if type(e[2]) == pd.DataFrame:
                        name = export_csv + '.' + e[0].replace(':', '-').replace(' ', '-') + '.csv'
                        e[2].to_csv(name, sep=",", index=False)
            file.write('\n')


def save_model(key, algorithm, conf):
    '''
    Save the model object for reuse with FileModel
        --------
        algorithm : object
            Dictionary of all results res[algorithm_key][metric_key]
        conf : object
            Configuration dictionary, has to include results.pickel_models
    '''

    file_name = conf['results']['folder'] + '/' + conf['key'] + '_' + conf['data']['name'] + '_' + key + '.pkl'
    file_name = Path(file_name)
    ensure_dir(file_name)
    file = open(file_name, 'wb')

    #pickle.dump(algorithm, file)
    dill.dump(algorithm, file)

    file.close()


def print_results(res):
    '''
    Print the result array
        --------
        res : dict
            Dictionary of all results res[algorithm_key][metric_key]
    '''
    for k, l in res.items():
        for e in l:
            print(k, ':', e[0], ' ', e[1])


def load_evaluation(module):
    '''
    Load the evaluation module
        --------
        module : string
            Just the last part of the path, e.g., evaluation_last
    '''
    return importlib.import_module('evaluation.' + module)


def create_algorithms_dict(list):
    '''
    Create algorithm instances from the list of algorithms in the configuration
        --------
        list : list of dicts
            Dicts represent a single algorithm with class, a key, and optionally a param dict
    '''

    algorithms = {}
    for algorithm in list:
        Class = load_class('algorithms.' + algorithm['class'])

        default_params = algorithm['params'] if 'params' in algorithm else {}
        random_params = generate_random_params(algorithm)
        params = {**default_params, **random_params}
        del default_params, random_params

        if 'params' in algorithm:
            if 'algorithms' in algorithm['params']:
                hybrid_algorithms = create_algorithms_dict(algorithm['params']['algorithms'])
                params['algorithms'] = []
                a_keys = []
                for k, a in hybrid_algorithms.items():
                    params['algorithms'].append(a)
                    a_keys.append(k)

        # instance = Class( **params )
        key = algorithm['key'] if 'key' in algorithm else algorithm['class']
        if 'params' in algorithm:
            if 'algorithms' in algorithm['params']:
                for k, val in params.items():
                    if k == 'algorithms':
                        for pKey in a_keys:
                            key += '-' + pKey
                    elif k == 'file':
                        key += ''
                    else:
                        key += '-'+str(k)+"="+str(val)
                        key = key.replace(',', '_')

            else:
                for k, val in params.items():
                    if k != 'file':
                        key += '-'+str(k)+"="+str(val)
                        key = key.replace(',', '_')
                    #key += '-' + '-'.join( map( lambda x: str(x[0])+'='+str(x[1]), params.items() ) )

        if 'params_var' in algorithm:
            for k, var in algorithm['params_var'].items():
                for val in var:
                    params[k] = val  # params.update({k: val})
                    kv = k
                    for v in val:
                        kv += '-'+str(v)
                    instance = Class(**params)
                    algorithms[key+kv] = instance
        else:
            instance = Class(**params)
            algorithms[key] = instance

    return algorithms


def generate_random_params(algorithm):
    params = {}

    if 'params_opt' in algorithm:
        for key, value in algorithm['params_opt'].items():
            space = []
            if type(value) == list:
                for entry in value:
                    if type(entry) == list:
                        space += entry
                        # space.append(entry)
                    elif type(entry) == dict:  # range
                        space += list(create_linspace(entry))
                    else:
                        space += [entry]
                        # space += entry
                chosen = random.choice(space)
            elif type(value) == dict:  # range
                if 'space' in value:
                    if value['space'] == 'weight':
                        space.append(create_weightspace(value))  # {from: 0.0, to: 0.9, in: 10, type: float}
                    elif value['space'] == 'recLen':
                        space.append(create_linspace(value))
                else:
                    space = create_linspace(value)  # {from: 0.0, to: 0.9, in: 10, type: float}
                chosen = random.choice(space)
                chosen = float(chosen) if 'type' in value and value['type'] == 'float' else chosen
            else:
                print('not the right type')

            params[key] = chosen

    return params


def create_weightspace(value):
    num = value['num']
    space = []
    sum = 0
    rand = 1
    for i in range(num-1):  # all weights excluding the last one
        while (sum+rand) >= 1:
            # rand = np.linspace(0, 1, num=0.05).astype('float32')
            rand = round(np.random.rand(), 2)
        space.append(rand)
        sum += rand
        rand = 1

    space.append(round(1-sum, 2))  # last weight
    return space


def create_linspace(value):
    start = value['from']
    end = value['to']
    steps = value['in']
    space = np.linspace(start, end, num=steps).astype(value['type'] if 'type' in value else 'float32')
    return space


def create_metric_list(list):
    '''
    Create metric class instances from the list of metrics in the configuration
        --------
        list : list of dicts
            Dicts represent a single metric with class and optionally the list length
    '''
    metrics = []
    for metric in list:
        metrics += create_metric(metric)

    return metrics


def create_metric(metric):
    metrics = []
    Class = load_class('evaluation.metrics.' + metric['class'])
    if 'length' in metric:
        for list_length in metric['length']:
            metrics.append(Class(list_length))
    else:
        metrics.append(Class())
    return metrics


def load_class(path):
    '''
    Load a class from the path in the configuration
        --------
        path : dict of dicts
            Path to the class, e.g., algorithms.knn.cknn.ContextKNNN
    '''
    module_name, class_name = path.rsplit('.', 1)

    Class = getattr(importlib.import_module(module_name), class_name)
    return Class


def ensure_dir(file_path):
    '''
    Create all directories in the file_path if non-existent.
        --------
        file_path : string
            Path to the a file
    '''
    directory = os.path.dirname(file_path)
    if not os.path.exists(directory):
        os.makedirs(directory)


def send_message(text):
    if NOTIFY:
        body = 'News from ' + socket.gethostname() + ': \n'
        body += text

        updater.bot.send_message(chat_id=CHAT_ID, text=body)


def send_exception(text):
    if NOTIFY:
        send_message(text)
        tmpfile = open('exception.txt', 'w')
        traceback.print_exc(file=tmpfile)
        tmpfile.close()
        send_file('exception.txt')


def send_file(file):
    if NOTIFY:
        file = open(file, 'rb')
        updater.bot.send_document(chat_id=CHAT_ID, document=file)
        file.close()


def status(bot, update):
    if NOTIFY:
        update.message.reply_text(
            'Running on {}'.format(socket.gethostname()))


if __name__ == '__main__':

    if len(sys.argv) > 1:
        main(sys.argv[1], out=sys.argv[2] if len(sys.argv) > 2 else None)
    else:
        main('conf/in', out='conf/out')
