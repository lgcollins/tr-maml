"""
Code modified from the code available at https://github.com/cbfinn/maml

Usage Instructions:
    5-shot sinusoid TR-MAML train:
        python main.py --datasource=sinusoid --logdir=logs/sine/ --log_number=50 --p_lr=0.00001 --TR_MAML=True   --metatrain_iterations=70000 --norm=None --update_batch_size=5 --train=True --resume=False

    5-shot sinusoid MAML train:
        python main.py --datasource=sinusoid --logdir=logs/sine/ --log_number=60 --TR_MAML=False --metatrain_iterations=70000 --norm=None --update_batch_size=5 --train=True --resume=False

    10-shot sinusoid TR-MAML train:
        python main.py --datasource=sinusoid --logdir=logs/sine/ --log_number=100 --p_lr=0.00002 --TR_MAML=True   --metatrain_iterations=70000 --norm=None --update_batch_size=10 --train=True --resume=False

    10-shot sinusoid MAML train:
        python main.py --datasource=sinusoid --logdir=logs/sine/ --log_number=110 --TR_MAML=False   --metatrain_iterations=70000 --norm=None --update_batch_size=10 --train=True --resume=False


    5-shot sinusoid TR-MAML test:
        python main.py --datasource=sinusoid --logdir=logs/sine/ --log_number=50 --p_lr=0.00001 --TR_MAML=True   --metatrain_iterations=70000 --norm=None --update_batch_size=5 --train=False --resume=True

    5-shot sinusoid MAML test:
        python main.py --datasource=sinusoid --logdir=logs/sine/ --log_number=60 --TR_MAML=False --metatrain_iterations=70000 --norm=None --update_batch_size=5 --train=False --resume=True

    10-shot sinusoid TR-MAML test:
        python main.py --datasource=sinusoid --logdir=logs/sine/ --log_number=100 --p_lr=0.00002 --TR_MAML=True   --metatrain_iterations=70000 --norm=None --update_batch_size=10 --train=False --resume=True

    10-shot sinusoid MAML test:
        python main.py --datasource=sinusoid --logdir=logs/sine/ --log_number=110 --TR_MAML=False   --metatrain_iterations=70000 --norm=None --update_batch_size=10 --train=False --resume=True

    Note that better sinusoid results can be achieved by using a larger network.
"""
import csv
import numpy as np
import pickle
import random
import math
# import tensorflow as tf
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()

from data_generator import DataGenerator
from maml import MAML
from tensorflow.python.platform import flags

FLAGS = flags.FLAGS

## Dataset/method options
flags.DEFINE_string('datasource', 'sinusoid', 'sinusoid or omniglot or miniimagenet')
flags.DEFINE_integer('num_classes', 5, 'number of classes used in classification (e.g. 5-way classification)')
# oracle means task id is input (only suitable for sinusoid)
flags.DEFINE_string('baseline', None, 'oracle, or None')
flags.DEFINE_bool('TR_MAML', False, 'True to use TR-MAML, False to use MAML')

## Training options
flags.DEFINE_integer('pretrain_iterations', 0, 'number of pre-training iterations.')
flags.DEFINE_integer('metatrain_iterations', 15000, 'number of metatraining iterations.') # 15k for omniglot, 50k for sinusoid
flags.DEFINE_integer('meta_batch_size', 25, 'number of tasks sampled per meta-update') # used to be 25
flags.DEFINE_float('meta_lr', 0.001, 'the base learning rate of the generator')
flags.DEFINE_integer('update_batch_size', 5, 'number of examples used for inner gradient update (K for K-shot learning).')
flags.DEFINE_float('update_lr', 1e-3, 'step size alpha for inner gradient update.') # 0.1 for omniglot
flags.DEFINE_float('p_lr', 0.0003, 'step size alpha for updating p.')
flags.DEFINE_integer('num_updates', 1, 'number of inner gradient updates during training.')

## Model options
flags.DEFINE_string('norm', 'batch_norm', 'batch_norm, layer_norm, or None')
flags.DEFINE_integer('num_filters', 64, 'number of filters for conv nets -- 32 for miniimagenet, 64 for omiglot.')
flags.DEFINE_bool('conv', True, 'whether or not to use a convolutional network, only applicable in some cases')
flags.DEFINE_bool('max_pool', False, 'Whether or not to use max pooling rather than strided convolutions')
flags.DEFINE_bool('stop_grad', False, 'if True, do not use second derivatives in meta-optimization (for speed)')

## Logging, saving, and testing options
flags.DEFINE_bool('log', False, 'if false, do not log summaries, for debugging code.')
flags.DEFINE_string('logdir', './sine_model_checkpts/', 'directory for summaries and checkpoints.')
flags.DEFINE_bool('resume', True, 'resume training if there is a model available') # was true earlier
flags.DEFINE_bool('train', True, 'True to train, False to test.')
flags.DEFINE_integer('test_iter', -1, 'iteration to load model (-1 for latest model)')
flags.DEFINE_bool('test_set', True, 'Set to true to test on the the test set, False for the validation set.')
flags.DEFINE_integer('train_update_batch_size', -1, 'number of examples used for gradient update during training (use if you want to test with a different number).')
flags.DEFINE_float('train_update_lr', -1, 'value of inner gradient step step during training. (use if you want to test with a different value)') # 0.1 for omniglot
flags.DEFINE_integer('log_number', 100, 'number to index saved models and results')

# The following code is adapted from the psuedocode in https://arxiv.org/pdf/1309.1541.pdf (Wang and Perpinan 2013)
def simplex_proj(beta):
    beta_sorted = np.flip(np.sort(beta))
    rho = 1
    for i in range(len(beta)-1):
        j = len(beta) - i
        test = beta_sorted[j-1] + (1 - np.sum(beta_sorted[:j]))/(j)
        if test > 0:
            rho = j
            break

    lam = (1-np.sum(beta_sorted[:rho]))/(rho)
    return np.maximum(beta + lam,0)

NUM_TEST_POINTS = 200

def trial(model, saver, sess, exp_string, data_generator, TR_MAML=False, resume_itr=0):
    random.seed(3)
    
    SUMMARY_INTERVAL = 10 
    # VAL_INTERVAL = 4
    SAVE_INTERVAL = 10000
    if FLAGS.datasource == 'sinusoid':
        PRINT_INTERVAL = 1000
        TEST_PRINT_INTERVAL = PRINT_INTERVAL
        VAL_INTERVAL = TEST_PRINT_INTERVAL
    else:
        PRINT_INTERVAL = 100
        TEST_PRINT_INTERVAL = PRINT_INTERVAL*5
        VAL_INTERVAL = TEST_PRINT_INTERVAL
        
    if not FLAGS.train:
        SUMMARY_INTERVAL = 1
        PRINT_INTERVAL = 1
        TEST_PRINT_INTERVAL = 201#PRINT_INTERVAL*1
        VAL_INTERVAL = 1

    if FLAGS.log:
        train_writer = tf.summary.FileWriter(FLAGS.logdir + '/' + exp_string, sess.graph)
    print('Done initializing, starting training. editted')
    postlosses = []
    valmean = 0
    valmax = 0
    valstd = 0
    val_losses = []
    
    num_tasks_mtrain = 100
    num_tasks_mtest = 490
    
    idx_counts = np.ones(num_tasks_mtrain)
    postloss_by_idx = np.zeros(num_tasks_mtrain)
    val_losses = np.zeros(num_tasks_mtest)
    val_idx_counts = np.ones(num_tasks_mtest)
    p = np.ones(num_tasks_mtrain)
    
    num_classes = data_generator.num_classes # for classification, 1 otherwise
    multitask_weights, reg_weights = [], []
    
    train_mean = np.zeros(250)
    train_max = np.zeros(250)
    val_mean = np.zeros(250)
    val_max = np.zeros(250)
    val_std = np.zeros(250)
    iters = np.zeros(250)
    iter_ind = 0
    
    counter=0
    val_means=[]
    val_maxs=[]
    val_stds=[]
    accs = []
    
    meta_learn_rate = FLAGS.meta_lr
    if not FLAGS.train:
        meta_learn_rate = 0
        
    if FLAGS.train:
        range_iters = range(resume_itr, FLAGS.pretrain_iterations + FLAGS.metatrain_iterations+1)
    else:
        range_iters = range(1)
    
    for itr in range_iters:
        feed_dict = {}
        if 'generate' in dir(data_generator):
            batch_x, batch_y, amp, phase, idx = data_generator.generate(train=True,val=False)

            if FLAGS.baseline == 'oracle':
                batch_x = np.concatenate([batch_x, np.zeros([batch_x.shape[0], batch_x.shape[1], 2])], 2)
                for i in range(FLAGS.meta_batch_size):
                    batch_x[i, :, 1] = amp[i]
                    batch_x[i, :, 2] = phase[i]

            inputa = batch_x[:, :num_classes*FLAGS.update_batch_size, :]
            labela = batch_y[:, :num_classes*FLAGS.update_batch_size, :]
            inputb = batch_x[:, num_classes*FLAGS.update_batch_size:, :] # b used for testing
            labelb = batch_y[:, num_classes*FLAGS.update_batch_size:, :]
            feed_dict = {model.inputa: inputa, model.inputb: inputb,  model.labela: labela, model.labelb: labelb, model.idx:idx, model.pweights:p,model.meta_lr:meta_learn_rate } #, model.meta_lr:0

        if itr < FLAGS.pretrain_iterations:
            input_tensors = [model.pretrain_op]
        else:
            input_tensors = [model.metatrain_op]
            # input_tensors = []
             
        if (itr % SUMMARY_INTERVAL == 0 or itr % PRINT_INTERVAL == 0 or 1):
#             input_tensors.extend([model.summ_op, model.total_loss1, model.total_losses2[FLAGS.num_updates-1]])
            input_tensors.extend([model.total_losses3, model.total_losses4, model.total_loss1, model.total_losses2[FLAGS.num_updates-1]])
            if model.classification:
                input_tensors.extend([model.total_accuracy1, model.total_accuracies2[FLAGS.num_updates-1]])

        result = sess.run(input_tensors, feed_dict)
        
        if (FLAGS.TR_MAML and FLAGS.train):
            real_p = p/num_tasks_mtrain
            for i in range(FLAGS.meta_batch_size):
                real_p[idx[i]] += FLAGS.p_lr*result[-3][0][i]   #0.00003 for K=10 ---- i really should do 0.00002 for da though
            # note that the stochastic gradient estimates have a factor of the number of tasks. We include that factor here in p.
            p = num_tasks_mtrain*simplex_proj(real_p)

        if itr % SUMMARY_INTERVAL == 0:
            if FLAGS.log:
                train_writer.add_summary(result[1], itr)
            postlosses.append(result[-1])
            for i in range(FLAGS.meta_batch_size):
                idx_counts[idx[i]] = math.floor(idx_counts[idx[i]]+1)
                postloss_by_idx[idx[i]] += result[-3][0][i]

        if itr % PRINT_INTERVAL == 0:
            if itr < FLAGS.pretrain_iterations:
                print_str = 'Pretrain Iteration ' + str(itr)
            else:
                print_str = 'Iteration ' + str(itr - FLAGS.pretrain_iterations)
            if itr == 0:
                train_max[iter_ind] = np.max(result[-3][0])
                train_mean[iter_ind] = np.mean(result[-3][0])
            else:
                postloss_by_idx = np.divide(postloss_by_idx, idx_counts)
                train_max[iter_ind] = np.max(postloss_by_idx)
                train_mean[iter_ind] = np.mean(postloss_by_idx)
            
            print_str += ': ' + str(train_max[iter_ind]) + ', ' + str(train_mean[iter_ind]) + ', ' + str(np.argmax(postloss_by_idx))
            print(print_str)
            postlosses = []
            postloss_by_idx = 0*postloss_by_idx
            idx_counts = 0.1*np.ones(num_tasks_mtrain)

        if (itr!=0) and itr % SAVE_INTERVAL == 0 and FLAGS.train:
            print(p)
            print("saving" + str(itr))
            saver.save(sess, FLAGS.logdir + '/' + exp_string + '/model' + str(itr))
        
        if itr % VAL_INTERVAL == 0:# and FLAGS.datasource !='sinusoid':
            for jj in range(NUM_TEST_POINTS):
                if 'generate' not in dir(data_generator):
                    feed_dict = {}
                    if model.classification:
                        input_tensors = [model.metaval_total_accuracy1, model.metaval_total_accuracies2[FLAGS.
                                                                          num_updates-1]]
                    else:
                        input_tensors = [model.metaval_total_loss1, model.metaval_total_losses2[FLAGS.num_updates-1] ]
                else:
                    batch_x, batch_y, amp, phase, idx = data_generator.generate(train=True, val=True)
                    inputa = batch_x[:, :num_classes*FLAGS.update_batch_size, :]
                    inputb = batch_x[:, num_classes*FLAGS.update_batch_size:, :]
                    labela = batch_y[:, :num_classes*FLAGS.update_batch_size, :]
                    labelb = batch_y[:, num_classes*FLAGS.update_batch_size:, :]
                    feed_dict = {model.inputa: inputa, model.inputb: inputb,  model.labela: labela, model.labelb: labelb, model.meta_lr: 0.0, model.idx:idx}
                    if model.classification:
                        input_tensors = [model.total_accuracy1, model.total_accuracies2[FLAGS.num_updates-1]]
                    else:
                        input_tensors = [model.outputbs[FLAGS.num_updates-1], model.total_losses4, model.total_loss1, model.total_losses2[FLAGS.num_updates-1]]
     
                result = sess.run(input_tensors, feed_dict)
                # pred = np.asarray(result[-4][0])
                # np.savetxt("gdrive/My Drive/maml/logs_sine/preds111.csv", pred, delimiter=",")
                valstd += np.std(result[-3][0])
                valmean += np.mean(result[-3][0])
                valmax += np.max(result[-3][0])
                
                for i in range(FLAGS.meta_batch_size):
                    val_idx_counts[idx[i]] = math.floor(val_idx_counts[idx[i]] + 1)
                    val_losses[idx[i]] += result[-3][0][i]
            
            # val_means.append(np.mean(result[-3][0]))
            # val_maxs.append(np.max(result[-3][0]))
            # val_stds.append(np.std(result[-3][0]))
            # accs.append(result[-3][0])

            if itr % TEST_PRINT_INTERVAL == 0 and FLAGS.train:

                if itr == 0:
                    val_max[iter_ind] = np.max(result[-3][0])
                    val_std[iter_ind] = np.std(result[-3][0])
                    val_mean[iter_ind] = np.mean(result[-3][0])
                else:
                    # val_max[iter_ind] = valmax
                    # val_std[iter_ind] = valstd
                    # val_mean[iter_ind] = valmean
                    
                    val_losses = np.divide(val_losses, val_idx_counts)
                    val_max[iter_ind] = np.max(val_losses)
                    val_std[iter_ind] = np.std(val_losses)
                    val_mean[iter_ind] = np.mean(val_losses)
                    
                print('Validation results: ' + str(val_max[iter_ind]) + ', ' + str(val_mean[iter_ind])+ ',' + str(np.argmax(val_losses)))
                iters[iter_ind] = itr
                iter_ind += 1
                valstd = 0
                valmean = 0
                valmax = 0
                val_losses = 0*val_losses
                val_idx_counts = 0.1*np.ones(490)
                
    if FLAGS.train:
        saver.save(sess, FLAGS.logdir + '/' + exp_string +  '/model' + str(itr))
        np.savetxt("logs_stats/train_mean" +str(FLAGS.log_number) + ".csv", train_mean, delimiter=",")
        np.savetxt("logs_stats/train_max"+str(FLAGS.log_number) + ".csv", train_max, delimiter=",")
        np.savetxt("logs_stats/val_mean"+str(FLAGS.log_number) + ".csv", val_mean, delimiter=",")
        np.savetxt("logs_stats/val_max"+str(FLAGS.log_number) + ".csv", val_max, delimiter=",")
        np.savetxt("logs_stats/val_std"+str(FLAGS.log_number) + ".csv", val_std, delimiter=",")
        np.savetxt("logs_stats/iters200.csv", iters, delimiter=",")
    else:
        val_losses = np.divide(val_losses, val_idx_counts)
        print('mean:')
        print(np.mean(val_losses))
        print('maxs:')
        print(np.max(val_losses))
        print('std:')
        print(np.std(val_losses))

        np.savetxt("logs_stats/test_accs"+str(FLAGS.log_number)+".csv", val_losses, delimiter=",")
    print("finish")

def main():
    if FLAGS.datasource == 'sinusoid':
        if FLAGS.train:
            test_num_updates = 1#5
        else:
            test_num_updates = 1#10
    else:
        if FLAGS.datasource == 'miniimagenet':
            if FLAGS.train == True:
                test_num_updates = 1  # eval on at least one update during training
            else:
                test_num_updates = 10
        else:
            test_num_updates = 10

    if FLAGS.train == False:
        orig_meta_batch_size = FLAGS.meta_batch_size

    if FLAGS.datasource == 'sinusoid':
        data_generator = DataGenerator(FLAGS.update_batch_size*2, FLAGS.meta_batch_size)
    else:
        if FLAGS.metatrain_iterations == 0 and FLAGS.datasource == 'miniimagenet':
            assert FLAGS.meta_batch_size == 1
            assert FLAGS.update_batch_size == 1
            data_generator = DataGenerator(1, FLAGS.meta_batch_size)  # only use one datapoint,
        else:
            if FLAGS.datasource == 'miniimagenet': # TODO - use 15 val examples for imagenet?
                if FLAGS.train:
                    data_generator = DataGenerator(FLAGS.update_batch_size+15, FLAGS.meta_batch_size)  # only use one datapoint for testing to save memory
                else:
                    data_generator = DataGenerator(FLAGS.update_batch_size*2, FLAGS.meta_batch_size)  # only use one datapoint for testing to save memory
            else:
                data_generator = DataGenerator(FLAGS.update_batch_size*2, FLAGS.meta_batch_size)  # only use one datapoint for testing to save memory

    dim_output = data_generator.dim_output
    if FLAGS.baseline == 'oracle':
        assert FLAGS.datasource == 'sinusoid'
        dim_input = 3
        FLAGS.pretrain_iterations += FLAGS.metatrain_iterations
        FLAGS.metatrain_iterations = 0
    else:
        dim_input = data_generator.dim_input

    if FLAGS.datasource == 'miniimagenet' or FLAGS.datasource == 'omniglot':
        tf_data_load = True
        num_classes = data_generator.num_classes

        if FLAGS.train: # only construct training model if needed
            random.seed(5)
            image_tensor, label_tensor = data_generator.make_data_tensor()
            inputa = tf.slice(image_tensor, [0,0,0], [-1,num_classes*FLAGS.update_batch_size, -1])
            inputb = tf.slice(image_tensor, [0,num_classes*FLAGS.update_batch_size, 0], [-1,-1,-1])
            labela = tf.slice(label_tensor, [0,0,0], [-1,num_classes*FLAGS.update_batch_size, -1])
            labelb = tf.slice(label_tensor, [0,num_classes*FLAGS.update_batch_size, 0], [-1,-1,-1])
            input_tensors = {'inputa': inputa, 'inputb': inputb, 'labela': labela, 'labelb': labelb}

        random.seed(6)
        image_tensor, label_tensor = data_generator.make_data_tensor(train=False)
        inputa = tf.slice(image_tensor, [0,0,0], [-1,num_classes*FLAGS.update_batch_size, -1])
        inputb = tf.slice(image_tensor, [0,num_classes*FLAGS.update_batch_size, 0], [-1,-1,-1])
        labela = tf.slice(label_tensor, [0,0,0], [-1,num_classes*FLAGS.update_batch_size, -1])
        labelb = tf.slice(label_tensor, [0,num_classes*FLAGS.update_batch_size, 0], [-1,-1,-1])
        metaval_input_tensors = {'inputa': inputa, 'inputb': inputb, 'labela': labela, 'labelb': labelb}
    else:
        tf_data_load = False
        input_tensors = None

    test_num_updates = 1
    model = MAML(dim_input, dim_output,test_num_updates=1)
    if FLAGS.train or not tf_data_load: 
        model.construct_model(input_tensors=input_tensors, prefix='metatrain_')
    if tf_data_load:
        model.construct_model(input_tensors=metaval_input_tensors, prefix='metaval_')
    # model.summ_op = tf.summary.merge_all()

    saver = loader = tf.train.Saver(tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES), max_to_keep=10)
    sess = tf.InteractiveSession()

    if FLAGS.train == False:
        # change to original meta batch size when loading model.
        FLAGS.meta_batch_size = orig_meta_batch_size

    if FLAGS.train_update_batch_size == -1:
        FLAGS.train_update_batch_size = FLAGS.update_batch_size
    if FLAGS.train_update_lr == -1:
        FLAGS.train_update_lr = FLAGS.update_lr

    # to test, resume must be true, and train must be false.
    exp_string = 'cls_'+str(FLAGS.num_classes)+ '.ubs_' + str(FLAGS.train_update_batch_size) + '.numstep' + str(FLAGS.num_updates) + '.updatelr' + str(FLAGS.train_update_lr) + '.iters'+ str(70000) + '.DA' + str(FLAGS.TR_MAML)+'.K'+str(FLAGS.update_batch_size)+ '.lognum'+ str(FLAGS.log_number)

    if FLAGS.num_filters != 64:
        exp_string += 'hidden' + str(FLAGS.num_filters)
    if FLAGS.max_pool:
        exp_string += 'maxpool'
    if FLAGS.stop_grad:
        exp_string += 'stopgrad'
    if FLAGS.baseline:
        exp_string += FLAGS.baseline
    if FLAGS.norm == 'batch_norm':
        exp_string += 'batchnorm'
    elif FLAGS.norm == 'layer_norm':
        exp_string += 'layernorm'
    # elif FLAGS.norm == 'None':
    #     exp_string += 'nonorm'
    # else:
    #     ('Norm setting not recognized.')

    resume_itr = 0
    model_file = None

    tf.global_variables_initializer().run()
    tf.train.start_queue_runners()

    print(exp_string)
    if FLAGS.resume or not FLAGS.train:
        model_file = tf.train.latest_checkpoint(FLAGS.logdir + '/' + exp_string)
        if FLAGS.test_iter > 0:
            model_file = model_file[:model_file.index('model')] + 'model' + str(FLAGS.test_iter)
        if model_file:
            ind1 = model_file.index('model')
            resume_itr = int(model_file[ind1+5:])
            print("Restoring model weights from " + model_file)
            saver.restore(sess, model_file)

    # run meta-training or meta-testing 
    trial(model, saver, sess, exp_string, data_generator, FLAGS.TR_MAML, resume_itr)
        
if __name__ == "__main__":
    main()
