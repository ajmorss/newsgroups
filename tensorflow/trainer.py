import tensorflow as tf
import numpy as np
from input_pipeline import get_input_iterator
from model import RNNModel
import argparse
import sys

from tensorflow.python.training.summary_io import SummaryWriterCache

training_filename = "../data/train.tfrecord"
test_filename = "../data/test.tfrecord"
# to do:
# synchronize training?


def validate_config(config):
    required_params = ['output_dir', 'epochs', 'batch_size', 'learning_rate']
    for x in required_params:
        if x not in config:
            raise "Invalid config: {} required".format(x)


def single_step(sess, chief_dict, step):
    res = sess.run(chief_dict['train_op'])
    chief_dict['summary_writer'].add_summary(res[1], step)
    chief_dict['summary_writer'].add_summary(res[-1], step)
    print('Step {}:'.format(str(res[2])))
    print('Train Loss: {}            Train Acc: {}'.format(str(res[3]), str(res[4])))
    return


def build_chief_graph(training_filename, test_filename):
    with tf.variable_scope('training_input_pipeline'):
        train_it, next_ele_train = get_input_iterator(training_filename,
                                                      config['batch_size'])
    with tf.variable_scope('test_input_pipeline'):
        test_it, next_ele_test = get_input_iterator(test_filename,
                                                    config['batch_size'])
    with tf.variable_scope("model"):
        model = RNNModel(next_ele_train, config)
    with tf.variable_scope("model", reuse=True):
        model_test = RNNModel(next_ele_test, config)
    output = {
                'next_ele_train': next_ele_train,
                'next_ele_test': next_ele_test,
                'model': model,
                'model_test': model_test
             }
    return output, train_it, test_it


def build_worker_graph(training_filename):
    with tf.variable_scope('training_input_pipeline'):
        it, next_ele = get_input_iterator(training_filename,
                                          config['batch_size'])
    with tf.variable_scope("model"):
        model = RNNModel(next_ele, config)
    return model, next_ele, it


def build_optimizer(loss):
    tvars = tf.trainable_variables()
    grads, _ = tf.clip_by_global_norm(tf.gradients(loss, tvars), 5)
    opt = tf.train.AdamOptimizer(config['learning_rate'])
    mini = opt.apply_gradients(zip(grads, tvars))
    return mini


def chief_session_setup():
    global_step = tf.Variable(0, trainable=False, name='global_step')
    update_step = tf.assign(global_step, global_step + 1)
    chief_dict, train_it, test_it = build_chief_graph(training_filename,
                                                      test_filename)

    loss = chief_dict['model'].loss_fn(chief_dict['next_ele_train'][2])

    mini = build_optimizer(loss)

    acc = chief_dict['model'].acc(chief_dict['next_ele_train'][2])
    test_acc, acc_ini = chief_dict['model_test'].stream_acc(chief_dict['next_ele_test'][2])

    train_sum = tf.summary.scalar("loss", loss)
    train_acc_sum = tf.summary.scalar("acc", acc)
    test_acc_sum = tf.summary.scalar("test_acc", test_acc)

    chief_dict['train_op'] = [mini, train_sum, update_step, loss, acc, train_acc_sum]
    chief_dict['test_op'] = [test_acc, test_acc_sum]

    return chief_dict, global_step, train_it, test_it, acc_ini


def chief_train(sess, chief_dict, global_step, train_it, test_it, acc_ini):
    chief_dict['summary_writer'] = SummaryWriterCache.get(config['output_dir'])
    sess.run(train_it.initializer)
    sess.run([test_it.initializer, acc_ini])
    for _ in xrange(config['epochs']):
        while True:
            try:
                step = sess.run(global_step)
                single_step(sess, chief_dict, step)
            except tf.errors.OutOfRangeError:
                break
        sess.run(train_it.initializer)
        while True:
            try:
                res = sess.run(chief_dict['test_op'])
            except tf.errors.OutOfRangeError:
                break
        chief_dict['summary_writer'].add_summary(res[-1], step)
        print('Test Acc: {}'.format(str(res[0])))
        sess.run([test_it.initializer, acc_ini])


def distributed_chief_run(server, cluster):
    with tf.device(tf.train.replica_device_setter(
                   worker_device="/job:worker/task:%d" % args.task_index,
                   cluster=cluster)):

        chief_dict, global_step, train_it, test_it, acc_ini = \
            chief_session_setup()

    # The MonitoredTrainingSession takes care of session initialization,
    # restoring from a checkpoint, saving to a checkpoint, and closing when
    # done or an error occurs.
    saver = tf.train.Saver(tf.global_variables(), max_to_keep=40)
    hooks = [tf.train.CheckpointSaverHook(config['output_dir'],
                                          save_secs=1800,
                                          save_steps=None,
                                          saver=saver,
                                          checkpoint_basename='model.ckpt')]
    sess = tf.train.MonitoredTrainingSession(master=server.target,
                                             is_chief=True,
                                             save_summaries_steps=None,
                                             save_summaries_secs=None,
                                             hooks=hooks)
    chief_train(sess, chief_dict, global_step, train_it, test_it, acc_ini)
    sess.close()


def local_chief_run():
    chief_dict, global_step, train_it, test_it, acc_ini = chief_session_setup()

    # The MonitoredTrainingSession takes care of session initialization,
    # restoring from a checkpoint, saving to a checkpoint, and closing when
    # done or an error occurs.
    saver = tf.train.Saver(tf.global_variables(), max_to_keep=40)
    hooks = [tf.train.CheckpointSaverHook(config['output_dir'],
                                          save_secs=1800,
                                          save_steps=None,
                                          saver=saver,
                                          checkpoint_basename='model.ckpt')]
    sess = tf.train.MonitoredTrainingSession(master="",
                                             checkpoint_dir=config['output_dir'],
                                             is_chief=True,
                                             save_summaries_steps=None,
                                             save_summaries_secs=None,
                                             hooks=hooks)
    chief_train(sess, chief_dict, global_step, train_it, test_it, acc_ini)
    sess.close()


def distributed_non_chief_run(server, cluster):
    with tf.device(tf.train.replica_device_setter(
                   worker_device="/job:worker/task:%d" % args.task_index,
                   cluster=cluster)):

        global_step = tf.Variable(0, trainable=False, name='global_step')
        update_step = tf.assign(global_step, global_step + 1)
        model, next_ele, it = build_worker_graph(training_filename)

        loss = model.loss_fn(next_ele[2])

        mini = build_optimizer(loss)
        train_op = [mini, update_step]

    sess = tf.train.MonitoredTrainingSession(master=server.target,
                                             is_chief=False,
                                             checkpoint_dir=config['output_dir'],
                                             save_summaries_steps=None,
                                             save_summaries_secs=None)
    for _ in xrange(config['epochs']):
        sess.run(it.initializer)
        while True:
            try:
                sess.run(train_op)
            except tf.errors.OutOfRangeError:
                break
    sess.close()


def main(_):
    if not args.distributed:
        local_chief_run()
        return
    ps_hosts = args.ps_hosts.split(",")
    worker_hosts = args.worker_hosts.split(",")

    # Create a cluster from the parameter server and worker hosts.
    cluster = tf.train.ClusterSpec({"ps": ps_hosts, "worker": worker_hosts})

    # Create and start a server for the local task.
    server = tf.train.Server(cluster,
                             job_name=args.job_name,
                             task_index=args.task_index)

    if args.job_name == "ps":
        server.join()
    elif args.job_name == "worker":
        is_chief = (args.task_index == 0)
        if is_chief:
            distributed_chief_run(server, cluster)
        else:
            distributed_non_chief_run(server, cluster)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.register("type", "bool", lambda v: v.lower() == "true")
    # Flags for defining the tf.train.ClusterSpec
    parser.add_argument(
        "--distributed",
        type=bool,
        default=False,
        help="Distributed"
    )
    parser.add_argument(
        "--ps_hosts",
        type=str,
        default="",
        help="Comma-separated list of hostname:port pairs"
    )
    parser.add_argument(
        "--worker_hosts",
        type=str,
        default="",
        help="Comma-separated list of hostname:port pairs"
    )
    parser.add_argument(
        "--job_name",
        type=str,
        default="",
        help="One of 'ps', 'worker'"
    )
    # Flags for defining the tf.train.Server
    parser.add_argument(
        "--task_index",
        type=int,
        default=0,
        help="Index of task within the job"
    )
    parser.add_argument(
        "--config_yaml",
        type=str,
        default='',
        help="Training configuration file"
    )
    args, unparsed = parser.parse_known_args()
    print(args)
    import yaml
    with open(args.config_yaml) as file_in:
        config = yaml.load(file_in)
        #config['l2_scale'] = float(config['l2_scale'])
        config['learning_rate'] = float(config['learning_rate'])
        print(config)
    tf.app.run(main=main, argv=[sys.argv[0]] + unparsed)
