import logging
import math
import numpy as np
import os
import random
import tensorflow as tf
import shutil

from data.mnist.read_mnist import MNIST
from data.isolet.read_isolet import Isolet
from data.omniglot.read_omniglot import Omniglot
from data.tiny_imagenet.read_tiny_imagenet import TinyImageNet
from data.aptos.read_aptos import Aptos
from data.episode_generator import generate_training_episode, generate_evaluation_episode 
from losses.histogram_loss import train_batches
from models.hist_model import *
from models.proto_model import *
from models.weight_transfer_model import *
from models.baseline_model import *
from models.recall_at_kappa import recall_at_kappa_leave_one_out, recall_at_kappa_support_query
from utils import classification_batch_evaluation, hist_loss_batch_eval, proto_episodic_performance, proto_performance

def train_classification(sess, model, data, params, weight_transfer=True):
    (x_train, y_train), (x_valid, y_valid), (x_train2, y_train2), (x_test2, y_test2) = data

    if weight_transfer:
        initial_best_epoch = {'epoch': -1, 'valid_acc': -1}

        for epoch in range(1, params['epochs'] + 1):

            shuffle = np.random.permutation(len(y_train))
            x_train, y_train = x_train[shuffle], y_train[shuffle]
            for i in range(0, len(y_train), params['batch_size']):
                x_train_mb, y_train_mb = x_train[i:i + params['batch_size']], y_train[i:i + params['batch_size']]
                sess.run(model.optimize, feed_dict={model.input: x_train_mb, model.target: y_train_mb, model.is_task1: True, model.is_train: True, model.learning_rate: params['learning_rate']})

            valid_acc = classification_batch_evaluation(sess, model, model.metrics, params['batch_size'], True, x_valid, y=y_valid, stream=True)

            print('valid [{} / {}] valid accuracy: {}'.format(epoch, params['epochs'] + 1, valid_acc))
            logging.info('valid [{} / {}] valid accuracy: {}'.format(epoch, params['epochs'] + 1, valid_acc))

            if valid_acc > initial_best_epoch['valid_acc']:
                initial_best_epoch['epoch'] = epoch
                initial_best_epoch['valid_acc'] = valid_acc
                model.save_model(sess, epoch)

            if epoch - initial_best_epoch['epoch'] >= params['patience']:
                print('Early Stopping Epoch: {}\n'.format(epoch))
                logging.info('Early Stopping Epoch: {}\n'.format(epoch))
                break

        print('Initial training done \n')
        logging.info('Initial training done \n')

        model.restore_model(sess)

    transfer_best_epoch = {'epoch': -1, 'train_acc': -1, 'test_acc': -1}
    es_acc = 0.0

    for epoch in range(1, params['epochs'] + 1):
        shuffle = np.random.permutation(len(y_train2))
        x_train2, y_train2 = x_train2[shuffle], y_train2[shuffle]
        for i in range(0, len(y_train2), params['batch_size']):
            x_train_mb, y_train_mb = x_train2[i:i + params['batch_size']], y_train2[i:i + params['batch_size']]
            sess.run(model.optimize, feed_dict={model.input: x_train_mb, model.target: y_train_mb, model.is_task1: False, model.is_train: True, model.learning_rate: params['learning_rate']})

        train_acc = classification_batch_evaluation(sess, model, model.metrics, params['batch_size'], False, x_train2, y=y_train2, stream=True)

        print('train [{} / {}] train accuracy: {}'.format(epoch, params['epochs'] + 1, train_acc))
        logging.info('train [{} / {}] train accuracy: {}'.format(epoch, params['epochs'] + 1, train_acc))

        if train_acc > transfer_best_epoch['train_acc']:
            transfer_best_epoch['epoch'] = epoch
            transfer_best_epoch['train_acc'] = train_acc
            test_acc = classification_batch_evaluation(sess, model, model.metrics, params['batch_size'], False, x_test2, y=y_test2, stream=True)
            transfer_best_epoch['test_acc'] = test_acc

        if epoch % params['patience'] == 0:
            acc_diff = transfer_best_epoch['train_acc'] - es_acc
            if acc_diff < params['percentage_es'] * es_acc:
                print('Early Stopping Epoch: {}\n'.format(epoch))
                logging.info('Early Stopping Epoch: {}\n'.format(epoch))
                break
            es_acc = transfer_best_epoch['train_acc']

    print('Transfer training done \n')
    print('test accuracy: {}'.format(transfer_best_epoch['test_acc']))
    logging.info('Transfer training done \n')
    logging.info('test accuracy: {}'.format(transfer_best_epoch['test_acc']))

def train_histogram_loss(sess, model, data, params):
    (x_train, y_train), (x_valid, y_valid), (x_train2, y_train2), (x_test2, y_test2) = data

    initial_best_epoch = {'epoch': -1, 'valid_acc': -1, 'test_acc': -1}

    for epoch in range(1, params['epochs'] + 1):
        shuffle = np.random.permutation(len(y_train))
        x_train, y_train = x_train[shuffle], y_train[shuffle]
        for batch_id, fd_items in train_batches(x_train, y_train, params['batch_size']):
            feed_dict = {
                model.input: fd_items['x'],
                model.pos_comps: fd_items['pos_comps'],
                model.neg_comps: fd_items['neg_comps'],
                model.n_pos_comps: fd_items['n_pos_comps'],
                model.n_neg_comps: fd_items['n_neg_comps'],
                model.is_train: True,
                model.learning_rate: params['learning_rate']
            }
            sess.run(model.optimize, feed_dict=feed_dict)

        valid_norm_preds = hist_loss_batch_eval(sess, model, model.prediction, params['batch_size'], x_valid)
        valid_recall_at_one = np.mean(recall_at_kappa_leave_one_out(valid_norm_preds, y_valid, kappa=params['kappa'], dist=params['dist']))

        print('valid [{} / {}] valid accuracy: {}'.format(epoch, params['epochs'] + 1, valid_recall_at_one))
        logging.info('valid [{} / {}] valid accuracy: {}'.format(epoch, params['epochs'] + 1, valid_recall_at_one))

        if valid_recall_at_one > initial_best_epoch['valid_acc']:
            initial_best_epoch['epoch'] = epoch
            initial_best_epoch['valid_acc'] = valid_recall_at_one

            if not params['adaptive'] or params['k'] <= 1:
                preds = hist_loss_batch_eval(sess, model, model.prediction, params['batch_size'], np.concatenate((x_train2, x_test2)))
                preds = np.split(preds, [len(x_train2), len(x_train2) + len(x_test2)])
                support_preds = preds[0]
                query_preds = preds[1]
                test_recall_at_one = np.mean(recall_at_kappa_support_query(support_preds, y_train2, query_preds, y_test2, kappa=params['kappa'], dist=params['dist']))
                initial_best_epoch['test_acc'] = test_recall_at_one

            model.save_model(sess, epoch)

        if epoch - initial_best_epoch['epoch'] >= params['patience']:
            print('Early Stopping Epoch: {}\n'.format(epoch))
            logging.info('Early Stopping Epoch: {}\n'.format(epoch))
            break

    if not params['adaptive'] or params['k'] <= 1:
        print('Optimization Finished \n')
        print('test accuracy: {}'.format(initial_best_epoch['test_acc']))
        logging.info('Optimization Finished \n')
        logging.info('test accuracy: {}'.format(initial_best_epoch['test_acc']))
        return

    print('Initial training done \n')
    logging.info('Initial training done \n')

    model.restore_model(sess)

    transfer_best_epoch = {'epoch': -1, 'train_acc': -1, 'test_acc': -1}
    es_acc = 0.0

    for epoch in range(1, params['epochs'] + 1):
        shuffle = np.random.permutation(len(x_train2))
        x_train2, y_train2 = x_train2[shuffle], y_train2[shuffle]
        for batch_id, fd_items in train_batches(x_train2, y_train2, params['batch_size']):
            feed_dict = {
                model.input: fd_items['x'],
                model.pos_comps: fd_items['pos_comps'],
                model.neg_comps: fd_items['neg_comps'],
                model.n_pos_comps: fd_items['n_pos_comps'],
                model.n_neg_comps: fd_items['n_neg_comps'],
                model.is_train: True,
                model.learning_rate: params['learning_rate']
            }
            sess.run(model.optimize, feed_dict=feed_dict)

        train_norm_preds = hist_loss_batch_eval(sess, model, model.prediction, params['batch_size'], x_train2)
        train_recall_at_one = np.mean(recall_at_kappa_leave_one_out(train_norm_preds, y_train2, kappa=params['kappa'], dist=params['dist']))

        print('train [{} / {}] train accuracy: {}'.format(epoch, params['epochs'] + 1, train_recall_at_one))
        logging.info('train [{} / {}] train accuracy: {}'.format(epoch, params['epochs'] + 1, train_recall_at_one))

        if train_recall_at_one > transfer_best_epoch['train_acc']:
            transfer_best_epoch['epoch'] = epoch
            transfer_best_epoch['train_acc'] = train_recall_at_one
            preds = hist_loss_batch_eval(sess, model, model.prediction, params['batch_size'], np.concatenate((x_train2, x_test2)))
            preds = np.split(preds, [len(x_train2), len(x_train2) + len(x_test2)])
            support_preds = preds[0]
            query_preds = preds[1]
            test_recall_at_one = np.mean(recall_at_kappa_support_query(support_preds, y_train2, query_preds, y_test2, kappa=params['kappa'], dist=params['dist']))
            transfer_best_epoch['test_acc'] = test_recall_at_one

        if epoch % params['patience'] == 0:
            acc_diff = transfer_best_epoch['train_acc'] - es_acc
            if acc_diff < params['percentage_es'] * es_acc:
                print('Early Stopping Epoch: {}\n'.format(epoch))
                logging.info('Early Stopping Epoch: {}\n'.format(epoch))
                break
            es_acc = transfer_best_epoch['train_acc']

    print('Transfer training done \n')
    print('test accuracy: {}'.format(transfer_best_epoch['test_acc']))
    logging.info('Transfer training done \n')
    logging.info('test accuracy: {}'.format(transfer_best_epoch['test_acc']))

def train_proto_nets(sess, model, data, params):
    (x_train, y_train), (x_valid, y_valid), (x_train2, y_train2), (x_test2, y_test2) = data

    # prepare tensorboard scalars
    valid_acc = 0
    train_acc = 0

    accuracy_summary = tf.Summary()

    # create writer
    if not os.path.exists("./graphs"):
        # shutil.rmtree("./graphs")
        os.mkdir("./graphs")
    writer = tf.summary.FileWriter('./graphs', sess.graph)

    model.create_saver()
    i = 1
    best_episode = {'episode': -1, 'valid_acc': -1, 'test_acc': -1}
    for support_batch, query_batch, query_labels_batch in generate_training_episode(x_train, y_train, 5, params['k'], params['query_train_per_class'], params['training_episodes'], batch_size=params['query_batch_size']):
    # for support_batch, query_batch, query_labels_batch in generate_training_episode(x_train, y_train, 5, 10, 2, 2, batch_size=params['query_batch_size']):
        # if i > 20000:
        #     break
        # auto adjust learning rate to avoid weight explosion
        if valid_acc < 0.45:
            feed_dict = {
                model.query: query_batch,
                model.label: query_labels_batch,
                model.is_train: True,
                model.learning_rate: params['learning_rate']
            }
        elif valid_acc < 0.5:
            feed_dict = {
                model.query: query_batch,
                model.label: query_labels_batch,
                model.is_train: True,
                model.learning_rate: params['learning_rate']/10
            }
        else:
            feed_dict = {
                model.query: query_batch,
                model.label: query_labels_batch,
                model.is_train: True,
                model.learning_rate: params['learning_rate']/10 * (1 - valid_acc)
            }

        if params['dataset'] == 'tiny_imagenet':
            prototypes = model.compute_batch_prototypes(sess, support_batch, 5)
            feed_dict[model.p] = prototypes
        else:
            feed_dict[model.support] = support_batch

        sess.run(model.optimize, feed_dict=feed_dict)

        # if i % 200 == 1:
        valid_cost, valid_acc = proto_performance(5, sess, model, x_train, y_train, x_valid, y_valid, 0)
        episode_support = math.floor(0.75 * params['k'])
        episode_query = params['k'] - episode_support
        valid_cost, valid_acc = float(valid_cost), float(valid_acc)
        print('valid [{}] valid cost: {} valid accuracy: {}'.format(i, valid_cost, valid_acc))
        logging.info('valid [{}] valid cost: {} valid accuracy: {}'.format(i, valid_cost, valid_acc))

        if i == 1:
            accuracy_summary.value.add(tag='task1_valid_accuracy', simple_value=0)
            accuracy_summary.value.add(tag='task1_valid_loss', simple_value=0)
        else:
            accuracy_summary.value[0].simple_value = valid_acc
            accuracy_summary.value[1].simple_value = valid_cost
            writer.add_summary(accuracy_summary)

        if valid_acc > best_episode['valid_acc']:
            best_episode['episode'] = i
            best_episode['valid_acc'] = valid_acc

            # if not params['adaptive']: # or params['k'] <= 1:
                # test_cost, test_acc = proto_performance(5, sess, model, x_train2, y_train2, x_test2, y_test2, batch_size=params['query_batch_size'])
                # best_episode['test_acc'] = float(test_acc)

            # model.save_model(sess, i)

        if i - best_episode['episode'] >= params['patience']:
            print('Early Stopping Episode: {}\n'.format(i))
            logging.info('Early Stopping Episode: {}\n'.format(i))
            break

        i += 1

    # if not params['adaptive']: # or params['k'] <= 1:
    print('Optimization Finished \n')
    # print('test accuracy: {}'.format(best_episode['test_acc']))
    logging.info('Optimization Finished \n')
    # logging.info('test accuracy: {}'.format(best_episode['test_acc']))
    return

    print('Initial training done \n')
    logging.info('Initial training done \n')

    model.save_model(sess, i)

    i = 1
    model.restore_model(sess)
    # Let 75% of the k points be used as support and rest as query when adapting
    episode_support = math.floor(0.75 * params['k'])
    episode_query = params['k'] - episode_support

    transfer_best_episode = {'episode': -1, 'train_acc': -1, 'test_acc': -1}
    es_acc = 0.0

    classes_per_episode = params['classes_per_episode']
    if params['dataset2'] is not None and params['dataset2'] == 'aptos':
            classes_per_episode = 5

    for support_batch, query_batch, query_labels_batch in generate_training_episode(x_train2, y_train2, classes_per_episode, episode_support, episode_query, params['training_episodes'], batch_size=params['query_batch_size']):
        feed_dict = {
            model.query: query_batch,
            model.label: query_labels_batch,
            model.is_train: True,
            model.learning_rate: params['learning_rate2']
        }

        # stop after 10000
        # if i > 20000:
        #     break
            
        if params['dataset2'] is not None:
            if params['dataset2'] == 'aptos':
                prototypes = model.compute_batch_prototypes(sess, support_batch, classes_per_episode)
                feed_dict[model.p] = prototypes
            else:
                feed_dict[model.support] = support_batch
        else:
            if params['dataset'] == 'tiny_imagenet':
                prototypes = model.compute_batch_prototypes(sess, support_batch, classes_per_episode)
                feed_dict[model.p] = prototypes
            else:
                feed_dict[model.support] = support_batch
        
        sess.run(model.optimize, feed_dict=feed_dict)

        if i % 200 == 1:
            train_perf, train_std = proto_episodic_performance(5, sess, model, x_train2, y_train2, classes_per_episode, episode_support, episode_query, params['query_batch_size'], params['evaluation_episodes'])
            train_perf[0] = float(train_perf[0])
            train_perf[1] = float(train_perf[1])
            print('train [{}] train cost: {} train accuracy: {}'.format(i, train_perf[1], train_perf[0]))
            logging.info('train [{}] train cost: {} train accuracy: {}'.format(i, train_perf[1], train_perf[0]))

            train_acc = train_perf[0]
            train_cost = train_perf[1]

            if i == 1:
                accuracy_summary.value.add(tag='task2_train_accuracy', simple_value=train_acc)
                accuracy_summary.value.add(tag='task2_train_loss', simple_value=train_cost)
            else:
                accuracy_summary.value[2].simple_value = train_acc
                accuracy_summary.value[3].simple_value = train_cost
                writer.add_summary(accuracy_summary)

            if train_perf[0] > transfer_best_episode['train_acc']:
                transfer_best_episode['episode'] = i
                transfer_best_episode['train_acc'] = train_perf[0]
                test_cost, test_acc = proto_performance(5, sess, model, x_train2, y_train2, x_test2, y_test2, batch_size=params['query_batch_size'])
                transfer_best_episode['test_acc'] = float(test_acc)

        if i % params['patience'] == 0:
            acc_diff = transfer_best_episode['train_acc'] - es_acc
            if acc_diff < params['percentage_es'] * es_acc:
                print('Early Stopping Episode: {}\n'.format(i))
                logging.info('Early Stopping Episode: {}\n'.format(i))
                break
            es_acc = transfer_best_episode['train_acc']

        i += 1

    print('Transfer training done \n')
    print('test accuracy: {}'.format(transfer_best_episode['test_acc']))
    logging.info('Transfer training done \n')
    logging.info('test accuracy: {}'.format(transfer_best_episode['test_acc']))

def get_model(params):
    model, data = None, None
    if params['command'] == 'hist_loss':
        if params['dataset'] == 'mnist':
            model = MNISTHistModel(params)
            data = MNIST(params['data_path']).kntl_data_form(params['t1_train'], params['t1_valid'], 
                params['k'], params['n'], params['t2_test'])
        elif params['dataset'] == 'isolet':
            model = IsoletHistModel(params)
            data = Isolet(params['data_path']).kntl_data_form(250, params['n'], params['k'], params['n'])
        elif params['dataset'] == 'omniglot':
            model = OmniglotHistModel(params)
            data = Omniglot(params['data_path']).kntl_data_form(params['n'], params['k'], params['n'])
        else:
            model = TinyImageNetHistModel(params)
            data = TinyImageNet(params['data_path'], params['data_path2']).kntl_data_form(350, params['n'], params['k'], params['n'], 2)
    elif params['command'] == 'proto':
        if params['dataset'] == 'mnist':
            model = MNISTProtoModel(params)
            data = MNIST(params['data_path']).kntl_data_form(params['t1_train'], params['t1_valid'], 
                params['k'], params['n'], params['t2_test'])
        elif params['dataset'] == 'isolet':
            model = IsoletProtoModel(params)
            data = Isolet(params['data_path']).kntl_data_form(250, params['n'], params['k'], params['n'])
        elif params['dataset'] == 'omniglot':
            model = OmniglotProtoModel(params)
            data = Omniglot(params['data_path']).kntl_data_form(params['n'], params['k'], params['n'])
        else:
            model = TinyImageNetProtoModel(params)
            # print(params['data_path2'])
            data = TinyImageNet(params['data_path'], params['data_path2']).kntl_data_form(100, 5, params['k'], 5, 3)
            # data = Aptos(params['data_path2']).kntl_data_form(5, 0)
    elif params['command'] == 'weight_transfer':
        if params['dataset'] == 'mnist':
            model = MNISTWeightTransferModel(params)
            data = MNIST(params['data_path']).kntl_data_form(params['t1_train'], params['t1_valid'], 
                params['k'], params['n'], params['t2_test'])
        elif params['dataset'] == 'isolet':
            model = IsoletWeightTransferModel(params)
            data = Isolet(params['data_path']).kntl_data_form(250, params['n'], params['k'], params['n'])
        elif params['dataset'] == 'omniglot':
            model = OmniglotWeightTransferModel(params)
            data = Omniglot(params['data_path']).kntl_data_form(params['n'], params['k'], params['n'])
        else:
            model = TinyImageNetWeightTransferModel(params)
            data = TinyImageNet(params['data_path'], params['data_path2']).kntl_data_form(350, params['n'], params['k'], params['n'], 2)
    elif params['command'] == 'baseline':
        if params['dataset'] == 'mnist':
            model = MNISTBaselineModel(params)
            data = MNIST(params['data_path']).kntl_data_form(params['t1_train'], params['t1_valid'], 
                params['k'], params['n'], params['t2_test'])
        elif params['dataset'] == 'isolet':
            model = IsoletBaselineModel(params)
            data = Isolet(params['data_path']).kntl_data_form(250, params['n'], params['k'], params['n'])
        elif params['dataset'] == 'omniglot':
            model = OmniglotBaselineModel(params)
            data = Omniglot(params['data_path']).kntl_data_form(params['n'], params['k'], params['n'])
        else:
            model = TinyImageNetBaselineModel(params)
            data = TinyImageNet(params['data_path'], params['data_path2']).kntl_data_form(350, params['n'], params['k'], 5, 2)
    else:
        print('Unknown model type')
        logging.debug('Unknown model type')
        quit()

    return model, data

def run(params):
    params = vars(params)
    logging.info(params)

    random.seed(params['random_seed'])
    initialization_seq = random.sample(range(50000), params['replications'])

    config = tf.ConfigProto(allow_soft_placement=True)
    config.gpu_options.allow_growth = True
    for rep in range(params['replications']):
        tf.reset_default_graph()
        with tf.Session(config=config) as sess:
            #tf.set_random_seed(initialization_seq[rep])
            np.random.seed(initialization_seq[rep])
            
            model, data = get_model(params)
            assert not model is None
            assert not data is None
            
            init = tf.group(tf.global_variables_initializer(), tf.local_variables_initializer())
            params['init'] = init
            model.create_saver()
            sess.run(init)
            
            rep_path = os.path.join(params['save_dir'], 'replication{}'.format(rep + 1))
            if not os.path.exists(rep_path):
            #     shutil.rmtree(rep_path)

                os.mkdir(rep_path)
            model.config['save_dir_by_rep'] = rep_path

            logging.debug('running training/testing')

            if params['command'] == 'baseline':
                train_classification(sess, model, data, params, weight_transfer=False)
            elif params['command'] == 'weight_transfer':
                train_classification(sess, model, data, params, weight_transfer=True)
            elif params['command'] == 'hist_loss':
                train_histogram_loss(sess, model, data, params)
            elif params['command'] == 'proto':
                train_proto_nets(sess, model, data, params)
            else:
                print('Unknown model type')
                logging.debug('Unknown model type')
                quit()
