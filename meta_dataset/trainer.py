# coding=utf-8
# Copyright 2019 The Meta-Dataset Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# Lint as: python2, python3
"""Interface for a learner that uses BenchmarkReaderDataSource to get data."""
# TODO(lamblinp): Update variable names to be more consistent
# - target, class_idx, label
# - support, query

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import collections
import os
import gin.tf
from meta_dataset import learner
from meta_dataset.data import dataset_spec as dataset_spec_lib
from meta_dataset.data import learning_spec
from meta_dataset.data import pipeline
from meta_dataset.data import providers
import numpy as np
import six
from six.moves import range
from six.moves import zip
import tensorflow as tf
from tqdm import tqdm

from tensorflow.core.protobuf import rewriter_config_pb2  # pylint: disable=g-direct-tensorflow-import

# Enable TensorFlow optimizations. It can add a few minutes to the first
# calls to session.run(), but decrease memory usage.
ENABLE_TF_OPTIMIZATIONS = True
# Enable tf.data optimizations, which are applied to the input data pipeline.
# It may be helpful to disable them when investigating regressions due to
# changes in tf.data (see b/121130181 for instance), but they seem to be helpful
# (or at least not detrimental) in general.
ENABLE_DATA_OPTIMIZATIONS = True

EMBEDDING_KEYWORDS = ('conv', 'resnet')

DATASETS_WITH_EXAMPLE_SPLITS = ()
TF_DATA_OPTIONS = tf.data.Options()
if not ENABLE_DATA_OPTIMIZATIONS:
  # The Options object can be used to control which static or dynamic
  # optimizations to apply.
  TF_DATA_OPTIONS.experimental_optimization.apply_default_optimizations = False


def make_summary(tag, val):
  '''
  Helper for manually creating a tensorboard summary
  '''
  summary = tf.Summary()
  summary.value.add(tag=tag, simple_value=val)
  return summary


def make_text_summary(tag, text):
  text_tensor = tf.make_tensor_proto(text, dtype=tf.string)
  meta = tf.SummaryMetadata()
  meta.plugin_data.plugin_name = "text"
  text_summary = tf.Summary()
  text_summary.value.add(tag=tag, metadata=meta, tensor=text_tensor)
  return text_summary


def get_uncertain_ratio(a, b, ci_a, ci_b, epsilon=1e-6):
  '''
  Compute CI of a/b assuming a and b have zero covariance
  https://en.wikipedia.org/wiki/Propagation_of_uncertainty#Example
  '''
  a = float(max(a, epsilon))
  b = float(max(b, epsilon))
  a_over_b = a / b
  ci_a_over_b = np.abs(a_over_b) * np.sqrt((ci_a/a)**2+(ci_b/b)**2)
  return a_over_b, ci_a_over_b


@gin.configurable('benchmark')
def get_datasets_and_restrictions(train_datasets='',
                                  eval_datasets='',
                                  restrict_num_per_class=None):
  """Gets the list of dataset names and possible restrictions on their classes.

  Args:
    train_datasets: A string of comma-separated dataset names for training.
    eval_datasets: A string of comma-separated dataset names for evaluation.
    restrict_num_per_class: If provided, a dict that maps dataset names to a
      dict that specifies for each of 'train', 'valid' and 'test' the number of
      examples per class to restrict to. For datasets / splits that are not
      specified, no restriction is applied.

  Returns:
    Two lists of dataset names and a possibly empty dictionary.
  """
  if restrict_num_per_class is None:
    restrict_num_per_class = {}

  train_datasets = [d.strip() for d in train_datasets.split(',')]
  eval_datasets = [d.strip() for d in eval_datasets.split(',')]

  return train_datasets, eval_datasets, restrict_num_per_class


def apply_dataset_options(dataset):
  """Apply the module-wide set of dataset options to dataset.

  In particular, this is used to enable or disable tf.data optimizations.
  This applies to the whole pipeline, so we can just set it at the end.

  Args:
    dataset: a tf.data.Dataset object.

  Returns:
    A tf.data.Dataset object with options applied.
  """
  return dataset.with_options(TF_DATA_OPTIONS)


def compute_train_class_proportions(episode, shots, dataset_spec):
  """Computes the proportion of each class' examples in the support set.

  Args:
    episode: An EpisodeDataset.
    shots: A 1D Tensor whose length is the `way' of the episode that stores the
      shots for this episode.
    dataset_spec: A DatasetSpecification.

  Returns:
    class_props: A 1D Tensor whose length is the `way' of the episode, storing
      for each class the proportion of its examples that are in the support set.
  """
  # Get the total number of examples of each class in the dataset.
  num_dataset_classes = len(dataset_spec.images_per_class)
  num_images_per_class = [
      dataset_spec.get_total_images_per_class(class_id)
      for class_id in range(num_dataset_classes)
  ]

  # Get the (absolute) class ID's that appear in the episode.
  class_ids, _ = tf.unique(episode.train_class_ids)  # [?, ]

  # Make sure that class_ids are valid indices of num_images_per_class. This is
  # important since tf.gather will fail silently and return zeros otherwise.
  num_classes = tf.shape(num_images_per_class)[0]
  check_valid_inds_op = tf.assert_less(class_ids, num_classes)
  with tf.control_dependencies([check_valid_inds_op]):
    # Get the total number of examples of each class that is in the episode.
    num_images_per_class = tf.gather(num_images_per_class, class_ids)  # [?, ]

  # Get the proportions of examples of each class that appear in the train set.
  class_props = tf.truediv(shots, num_images_per_class)
  return class_props


def get_split_enum(split):
  """Returns the Enum value corresponding to the given split.

  Args:
    split: A String.

  Raises:
    ValueError: Split must be one of 'train', 'valid' or 'test'.
  """
  # Get the int representing the chosen split.
  if split == 'train':
    split_enum = learning_spec.Split.TRAIN
  elif split == 'valid':
    split_enum = learning_spec.Split.VALID
  elif split == 'test':
    split_enum = learning_spec.Split.TEST
  else:
    raise ValueError('Split must be one of "train", "valid" or "test".')
  return split_enum


def compute_episode_stats(episode):
  """Computes various episode stats: way, shots, and class IDs.

  Args:
    episode: An EpisodeDataset.

  Returns:
    way: An int constant tensor. The number of classes in the episode.
    shots: An int 1D tensor: The number of support examples per class.
    class_ids: An int 1D tensor: (absolute) class IDs.
  """
  # The train labels of the next episode.
  train_labels = episode.train_labels
  # Compute way.
  episode_classes, _ = tf.unique(train_labels)
  way = tf.size(episode_classes)
  # Compute shots.
  class_ids = tf.reshape(tf.range(way), [way, 1])
  class_labels = tf.reshape(train_labels, [1, -1])
  is_equal = tf.equal(class_labels, class_ids)
  shots = tf.reduce_sum(tf.cast(is_equal, tf.int32), axis=1)
  # Compute class_ids.
  class_ids, _ = tf.unique(episode.train_class_ids)
  return way, shots, class_ids


@gin.configurable
class LearnConfig(object):
  """A collection of values pertaining to learning."""

  def __init__(self, num_updates, batch_size, num_eval_episodes,
               checkpoint_every, validate_every, log_every,
               transductive_batch_norm,
               num_eval_other_metrics=0):
    """Initializes a LearnConfig.

    Args:
      num_updates: An integer, the number of training updates.
      batch_size: An integer, the size of batches for non-episodic models.
      num_eval_episodes: An integer, the number of episodes for evaluation.
      checkpoint_every: An integer, the number of episodes between consecutive
        checkpoints.
      validate_every: An integer, the number of episodes between consecutive
        validatations.
      log_every: An integer, the number of episodes between consecutive logging.
      transductive_batch_norm: Whether to batch normalize in the transductive
        setting where the mean and variance for normalization are computed from
        both the support and query sets.
    """
    self.num_updates = num_updates
    self.batch_size = batch_size
    self.num_eval_episodes = num_eval_episodes
    self.checkpoint_every = checkpoint_every
    self.validate_every = validate_every
    self.log_every = log_every
    self.transductive_batch_norm = transductive_batch_norm
    self.num_eval_other_metrics = num_eval_other_metrics


# TODO(manzagop): consider moving here num_updates/batch_size from LearnConfig.
# TODO(manzagop): consider combining train and eval learner and inferring
#   episodic.
@gin.configurable
class LearnerConfig(object):
  """A collection of values pertaining to the model."""

  def __init__(
      self,
      episodic,
      train_learner,
      eval_learner,
      pretrained_checkpoint,
      checkpoint_for_eval,
      embedding_network,
      learning_rate,
      decay_learning_rate,
      decay_every,
      decay_rate,
      experiment_name,
      pretrained_source,
  ):
    # pyformat: disable
    """Initializes a LearnerConfig.

    Args:
      episodic: A boolean, whether meta-training is episodic.
      train_learner: A string, the name of the learner to use for meta-training.
      eval_learner: A string, the name of the learner to use for
        meta-evaluation.
      pretrained_checkpoint: A string, the path to a checkpoint to use for
        initializing a model prior to training.
      checkpoint_for_eval: A string, the path to a checkpoint to restore for
        evaluation.
      embedding_network: A string, the embedding network to use.
      learning_rate: A float, the meta-learning learning rate.
      decay_learning_rate: A boolean, whether to decay the learning rate.
      decay_every: An integer, the learning rate is decayed for every multiple
        of this value.
      decay_rate: A float, the decay to apply to the learning rate.
      experiment_name: A string, a name for the experiment.
      pretrained_source: A string, the pretraining setup to use.
    """
    # pyformat: enable
    self.episodic = episodic
    self.train_learner = train_learner
    self.eval_learner = eval_learner
    self.pretrained_checkpoint = pretrained_checkpoint
    self.checkpoint_for_eval = checkpoint_for_eval
    self.embedding_network = embedding_network
    self.learning_rate = learning_rate
    self.decay_learning_rate = decay_learning_rate
    self.decay_every = decay_every
    self.decay_rate = decay_rate
    self.experiment_name = experiment_name
    self.pretrained_source = pretrained_source


@gin.configurable
class Trainer(object):
  """A Trainer for training a Learner on data provided by ReaderDataSource."""

  def __init__(
      self,
      train_learner,
      eval_learner,
      is_training,
      train_dataset_list,
      eval_dataset_list,
      restrict_num_per_class,
      checkpoint_dir,
      summary_dir,
      records_root_dir,
      eval_finegrainedness,
      eval_finegrainedness_split,
      eval_imbalance_dataset,
      omit_from_saving_and_reloading,
      train_episode_config,
      eval_episode_config,
      learn_config,
      learner_config,
      data_config,
  ):
    """Initializes a Trainer.

    Args:
      train_learner: A Learner to be used for meta-training.
      eval_learner: A Learner to be used for meta-validation or meta-testing.
      is_training: Bool, whether or not to train or just evaluate.
      train_dataset_list: A list of names of datasets to train on. This can be
        any subset of the supported datasets.
      eval_dataset_list: A list of names of datasets to evaluate on either for
        validation during train or for final test evaluation, depending on the
        nature of the experiment, as dictated by `is_training'.
      restrict_num_per_class: A dict that maps dataset names to a dict that
        specifies for each of 'train', 'valid' and 'test' the number of examples
        per class to restrict to. For datasets / splits that are not mentioned,
        no restriction is applied. If restrict_num_per_class is the empty dict,
        no restriction is applied to any split of any dataset.
      checkpoint_dir: A string, the path to the checkpoint directory, or None if
        no checkpointing should occur.
      summary_dir: A string, the path to the checkpoint directory, or None if no
        summaries should be saved.
      records_root_dir: A string, the path to the dataset records directory.
      eval_finegrainedness: Whether to perform binary ImageNet evaluation for
        assessing the performance on fine- vs coarse- grained tasks.
      eval_finegrainedness_split: The subgraph of ImageNet to perform the
        aforementioned analysis on. Notably, if this is 'train', we need to
        ensure that an training data is used episodically, even if the given
        model is the baseline model which usually uses batches for training.
      eval_imbalance_dataset: A dataset on which to perform evaluation for
        assessing how class imbalance affects performance in binary episodes. By
        default it is empty and no imbalance analysis is performed.
      omit_from_saving_and_reloading: A list of strings that specifies
        substrings of variable names that should not be reloaded.
      train_episode_config: An instance of EpisodeDescriptionConfig (in
        data/config.py). This is a config for setting the ways and shots of
        training episodes or the parameters for sampling them, if variable.
      eval_episode_config: An instance of EpisodeDescriptionConfig. Analogous to
        train_episode_config but used for eval episodes (validation or testing).
      learn_config: A LearnConfig, the learning configuration.
      learner_config: A LearnerConfig, the learner configuration.
      data_config: A DataConfig, the data configuration.
    """
    self.train_learner_class = train_learner
    self.eval_learner_class = eval_learner
    self.is_training = is_training
    self.train_dataset_list = train_dataset_list
    self.eval_dataset_list = eval_dataset_list
    self.restrict_num_per_class = restrict_num_per_class
    self.checkpoint_dir = checkpoint_dir
    self.summary_dir = summary_dir
    self.records_root_dir = records_root_dir
    self.eval_finegrainedness = eval_finegrainedness
    self.eval_finegrainedness_split = eval_finegrainedness_split
    self.eval_imbalance_dataset = eval_imbalance_dataset

    self.eval_split = 'test'
    if eval_finegrainedness:
      # The fine- vs coarse- grained evaluation may potentially be performed on
      # the training graph as it exhibits greater variety in this aspect.
      self.eval_split = eval_finegrainedness_split

    if eval_finegrainedness or eval_imbalance_dataset:
      # We restrict this analysis to the binary classification setting.
      tf.logging.info(
          'Forcing the number of {} classes to be 2, since '
          'the finegrainedness analysis is applied on binary '
          'classification tasks only.'.format(eval_finegrainedness_split))
      if eval_finegrainedness and eval_finegrainedness_split == 'train':
        train_episode_config.num_ways = 2
      else:
        eval_episode_config.num_ways = 2

    self.num_train_classes = train_episode_config.num_ways
    self.num_test_classes = eval_episode_config.num_ways
    self.num_support_train = train_episode_config.num_support
    self.num_query_train = train_episode_config.num_query
    self.num_support_eval = eval_episode_config.num_support
    self.num_query_eval = eval_episode_config.num_query

    self.learn_config = learn_config
    self.learner_config = learner_config
    self.train_episode_config = train_episode_config
    self.eval_episode_config = eval_episode_config

    if self.learn_config.transductive_batch_norm:
      tf.logging.warn('Using transductive batch norm!')

    # Only applicable for non-transudctive batch norm. The correct
    # implementation here involves computing the mean and variance based on the
    # support set and then using them to batch normalize the query set. During
    # meta-learning, we allow the gradients to flow through those moments.
    self.backprop_through_moments = True

    self.data_config = data_config
    # Get the image shape.
    self.image_shape = [data_config.image_height] * 2 + [3]

    # Create the benchmark specification.
    self.benchmark_spec = self.get_benchmark_specification()

    # Which splits to support depends on whether we are in the meta-training
    # phase or not. If we are, we need the train split, and the valid one for
    # early-stopping. If not, we only need the test split.
    if self.is_training:
      self.required_splits = ['train', 'valid']
    else:
      self.required_splits = [self.eval_split]

    # Get the training, validation and testing specifications.
    # Each is either an EpisodeSpecification or a BatchSpecification.
    split_episode_or_batch_specs = {}
    if 'train' in self.required_splits:
      split_episode_or_batch_specs['train'] = self._create_train_specification()
    for split in ['valid', 'test']:
      if split not in self.required_splits:
        continue
      split_episode_or_batch_specs[split] = self._create_held_out_specification(
          split)
    self.split_episode_or_batch_specs = split_episode_or_batch_specs

    # Get the next data (episode or batch) for the different splits.
    self.next_data = {}
    for split in self.required_splits:
      self.next_data[split] = self.build_data(split)

    # Initialize the learners.
    self.ema_object = None  # Using dummy EMA object for now.
    self.learners = {}
    self.embedding_fn = learner.NAME_TO_EMBEDDING_NETWORK[
        self.learner_config.embedding_network]
    if 'train' in self.required_splits:
      self.learners['train'] = (
          self.create_train_learner(self.train_learner_class,
                                    self.get_next('train')))
    if self.eval_learner_class is not None:
      for split in ['valid', 'test']:
        if split not in self.required_splits:
          continue
        self.learners[split] = self.create_eval_learner(self.eval_learner_class,
                                                        self.get_next(split))

    # Get the Tensors for the losses / accuracies of the different learners.
    self.losses = dict(
        zip(self.required_splits, [
            self.learners[split].compute_loss()
            for split in self.required_splits
        ]))
    self.accs = dict(
        zip(self.required_splits, [
            self.learners[split].compute_accuracy()
            for split in self.required_splits
        ]))

    # CentroidNetworkLearner has other metrics; return a list of metrics
    self.metrics = dict(
        zip(self.required_splits, [
            self.learners[split].get_metrics()
            for split in self.required_splits
        ]))

    # CentroidNetworkLearner needs to return embeddings
    self.tensors_for_metrics = dict(
        zip(self.required_splits, [
            self.learners[split].get_tensors_for_metrics()
            for split in self.required_splits
        ]))

    # Set self.way, self.shots to Tensors for the way/shots of the next episode.
    self.set_way_shots_classes_logits_targets()

    # Get an optimizer and the operation for meta-training.
    self.train_op = None
    if self.is_training:
      global_step = tf.train.get_or_create_global_step()
      learning_rate = self.learner_config.learning_rate
      if self.learner_config.decay_learning_rate:
        learning_rate = tf.train.exponential_decay(
            self.learner_config.learning_rate,
            global_step,
            decay_steps=self.learner_config.decay_every,
            decay_rate=self.learner_config.decay_rate,
            staircase=True)
      tf.summary.scalar('learning_rate', learning_rate)
      self.optimizer = tf.train.AdamOptimizer(learning_rate)
      self.train_op = self.get_train_op(global_step)

    vars_to_restore = []
    # Omit from reloading any variables that contains as a substring anything in
    # the following list. For example, those that track iterator state, as
    # iterator state is not saved.
    tf.logging.info('Omitting from saving / reloading any variable that '
                    'contains any of the following substrings: %s' %
                    omit_from_saving_and_reloading)
    for var in tf.global_variables():
      if not any([
          substring in var.name for substring in omit_from_saving_and_reloading
      ]):
        vars_to_restore.append(var)
      else:
        tf.logging.info('Omitting variable %s' % var.name)
    self.saver = tf.train.Saver(var_list=vars_to_restore, max_to_keep=500)

    if self.checkpoint_dir is not None:
      if not tf.gfile.Exists(self.checkpoint_dir):
        tf.gfile.MakeDirs(self.checkpoint_dir)

    # Initialize a Session.
    self.initialize_session()
    self.create_summary_writer()

  def set_way_shots_classes_logits_targets(self):
    """Sets the Tensors for the above info of the learner's next episode.

    Raises:
      NotImplementedError: Abstract Method.
    """
    raise NotImplementedError('Abstract Method.')

  def maybe_set_way_shots_classes_logits_targets(self, skip_train=False):
    """Wrapper for set_way_shots_classes_logits_targets.

    Args:
      skip_train: Whether to assign each of the way, shot, (absolute) class id,
        test_logits and test_targets of the training split to None. This is used
        when called with the training split from the batch trainer, since
        training does not happen in episodes there.
    """
    # The batch trainer receives episodes only for the valid and test splits.
    # Therefore for the train split there is no defined way and shots.
    (way, shots, class_props, class_ids, test_logits,
     test_targets) = [], [], [], [], [], []
    for split in self.required_splits:
      if split == 'train' and skip_train:
        (way_, shots_, class_props_, class_ids_, test_logits_,
         test_targets_) = [None] * 6
      else:
        way_, shots_, class_ids_ = compute_episode_stats(self.next_data[split])
        class_props_ = None
        if self.eval_imbalance_dataset:
          class_props_ = compute_train_class_proportions(
              self.next_data[split], shots_, self.eval_imbalance_dataset_spec)
        test_logits_ = self.learners[split].test_logits
        test_targets_ = self.learners[split].test_targets
      way.append(way_)
      shots.append(shots_)
      class_props.append(class_props_)
      class_ids.append(class_ids_)
      test_logits.append(test_logits_)
      test_targets.append(test_targets_)

    self.way = dict(zip(self.required_splits, way))
    self.shots = dict(zip(self.required_splits, shots))
    self.class_props = dict(zip(self.required_splits, class_props))
    self.class_ids = dict(zip(self.required_splits, class_ids))
    self.test_logits = dict(zip(self.required_splits, test_logits))
    self.test_targets = dict(zip(self.required_splits, test_targets))

  def create_summary_writer(self):
    """Create summaries and writer."""
    # Add summaries for the losses / accuracies of the different learners.
    standard_summaries = []
    for split in self.required_splits:
      loss_summary = tf.summary.scalar('%s_loss' % split, self.losses[split])
      acc_summary = tf.summary.scalar('%s_acc' % split, self.accs[split])
      standard_summaries.append(loss_summary)
      standard_summaries.append(acc_summary)

      # add other metrics
      for metric_name, metric_tensor in self.metrics[split].items():
        metric_summary = tf.summary.scalar('{}_{}'.format(split, metric_name), metric_tensor)
        standard_summaries.append(metric_summary)

    # Add summaries for the way / shot / logits / targets of the learners.
    evaluation_summaries = self.add_eval_summaries()

    # All summaries.
    self.standard_summaries = tf.summary.merge(standard_summaries)
    self.evaluation_summaries = tf.summary.merge(evaluation_summaries)

    # Get a writer.
    self.summary_writer = None
    if self.summary_dir is not None:
      self.summary_writer = tf.summary.FileWriter(self.summary_dir)
      if not tf.gfile.Exists(self.summary_dir):
        tf.gfile.MakeDirs(self.summary_dir)

  def create_train_learner(self, train_learner_class, episode_or_batch):
    """Instantiates a train learner.

    Args:
      train_learner_class: A train learner subclass of the Learner class.
      episode_or_batch: An EpisodeDataset or Batch.

    Raises:
      NotImplementedError: Abstract Method.
    """
    raise NotImplementedError('Abstract Method.')

  def create_eval_learner(self, eval_learner_class, episode):
    """Instantiates an eval learner.

    Args:
      eval_learner_class: An eval learner subclass of the Learner class.
      episode: An EpisodeDataset.

    Raises:
      NotImplementedError: Abstract Method.
    """
    raise NotImplementedError('Abstract Method.')

  def get_benchmark_specification(self, records_root_dir=None):
    """Returns a BenchmarkSpecification.

    Args:
      records_root_dir: Optional. If provided, a list or string that sets the
        directory in which a child directory will be searched for each dataset
        to locate that dataset's records and dataset specification. If it's a
        string, that path will be used for all datasets. If it's a list, its
        length must be the same as the number of datasets, in order to specify a
        different such directory for each. If None, self.records_root_dir will
        be used for all datasets.

    Raises:
      RuntimeError: Incorrect file_pattern detected in a dataset specification.
    """
    (data_spec_list, has_dag_ontology, has_bilevel_ontology,
     splits_to_contribute) = [], [], [], []
    seen_datasets = set()

    if self.is_training:
      benchmark_datasets = self.train_dataset_list + self.eval_dataset_list
    else:
      benchmark_datasets = self.eval_dataset_list

    if isinstance(records_root_dir, list):
      if len(records_root_dir) != len(benchmark_datasets):
        raise ValueError('The given records_root_dir is a list whose length is '
                         'not the same as the number of benchmark datasets. '
                         'Found datasets {} (for the {} phase) but '
                         'len(records_root_dir) is {}. Expected their lengths '
                         'to match or records_path to be a string').format(
                             benchmark_datasets, len(records_root_dir))
      records_roots_for_datasets = records_root_dir
    elif isinstance(records_root_dir, six.text_type):
      records_roots_for_datasets = [records_root_dir] * len(benchmark_datasets)
    elif records_root_dir is None:
      records_roots_for_datasets = [self.records_root_dir
                                   ] * len(benchmark_datasets)

    for dataset_name, dataset_records_root in zip(benchmark_datasets,
                                                  records_roots_for_datasets):

      # Might be seeing a dataset for a second time if it belongs to both the
      # train and eval dataset lists.
      if dataset_name in seen_datasets:
        continue

      dataset_records_path = os.path.join(dataset_records_root, dataset_name)
      data_spec = dataset_spec_lib.load_dataset_spec(dataset_records_path)

      # Only ImageNet has a DAG ontology.
      has_dag = (dataset_name == 'ilsvrc_2012')
      # Only Omniglot has a bi-level ontology.
      is_bilevel = (dataset_name == 'omniglot')

      # The meta-splits that this dataset will contribute data to.
      if not self.is_training:
        # If we're meta-testing, all datasets contribute only to meta-test.
        splits = {'test'}
      else:
        splits = set()
        if dataset_name in self.train_dataset_list:
          splits.add('train')
        if dataset_name in self.eval_dataset_list:
          splits.add('valid')

      # Add this dataset to the benchmark.
      tf.logging.info('Adding dataset {}'.format(data_spec.name))
      data_spec_list.append(data_spec)
      has_dag_ontology.append(has_dag)
      has_bilevel_ontology.append(is_bilevel)
      splits_to_contribute.append(splits)

      # Book-keeping.
      seen_datasets.add(dataset_name)

    if self.eval_imbalance_dataset:
      self.eval_imbalance_dataset_spec = data_spec
      assert len(data_spec_list) == 1, ('Imbalance analysis is only '
                                        'supported on one dataset at a time.')

    benchmark_spec = dataset_spec_lib.BenchmarkSpecification(
        'benchmark', self.image_shape, data_spec_list, has_dag_ontology,
        has_bilevel_ontology, splits_to_contribute)

    # Logging of which datasets will be used for the different meta-splits.
    splits_to_datasets = collections.defaultdict(list)
    for dataset_spec, splits_to_contribute in zip(data_spec_list,
                                                  splits_to_contribute):
      for split in splits_to_contribute:
        splits_to_datasets[split].append(dataset_spec.name)
    for split, datasets in splits_to_datasets.items():
      tf.logging.info('Episodes for split {} will be created from {}'.format(
          split, datasets))

    return benchmark_spec

  def initialize_session(self):
    """Initializes a tf Session."""
    if ENABLE_TF_OPTIMIZATIONS:
      self.sess = tf.Session()
    else:
      rewriter_config = rewriter_config_pb2.RewriterConfig(
          disable_model_pruning=True,
          constant_folding=rewriter_config_pb2.RewriterConfig.OFF,
          arithmetic_optimization=rewriter_config_pb2.RewriterConfig.OFF,
          remapping=rewriter_config_pb2.RewriterConfig.OFF,
          shape_optimization=rewriter_config_pb2.RewriterConfig.OFF,
          dependency_optimization=rewriter_config_pb2.RewriterConfig.OFF,
          function_optimization=rewriter_config_pb2.RewriterConfig.OFF,
          layout_optimizer=rewriter_config_pb2.RewriterConfig.OFF,
          loop_optimization=rewriter_config_pb2.RewriterConfig.OFF,
          memory_optimization=rewriter_config_pb2.RewriterConfig.NO_MEM_OPT)
      graph_options = tf.GraphOptions(rewrite_options=rewriter_config)
      session_config = tf.ConfigProto(graph_options=graph_options)
      self.sess = tf.Session(config=session_config)

    # Restore or initialize the variables.
    self.sess.run(tf.global_variables_initializer())
    self.sess.run(tf.local_variables_initializer())
    if self.learner_config.checkpoint_for_eval:
      # Requested a specific checkpoint.
      self.saver.restore(self.sess, self.learner_config.checkpoint_for_eval)
      tf.logging.info('Restored checkpoint: %s' %
                      self.learner_config.checkpoint_for_eval)
    else:
      # Continue from the latest checkpoint if one exists.
      # This handles fault-tolerance.
      latest_checkpoint = None
      if self.checkpoint_dir is not None:
        latest_checkpoint = tf.train.latest_checkpoint(self.checkpoint_dir)
      if latest_checkpoint:
        self.saver.restore(self.sess, latest_checkpoint)
        tf.logging.info('Restored checkpoint: %s' % latest_checkpoint)
      else:
        tf.logging.info('No previous checkpoint.')
        self.sess.run(tf.global_variables_initializer())
        self.sess.run(tf.local_variables_initializer())

    # For episodic models, potentially use pretrained weights at the start of
    # training. If this happens it will overwrite the embedding weights, but
    # taking care to not restore the Adam parameters.
    if self.learner_config.pretrained_checkpoint and not self.sess.run(
        tf.train.get_global_step()):
      self.saver.restore(self.sess, self.learner_config.pretrained_checkpoint)
      tf.logging.info('Restored checkpoint: %s' %
                      self.learner_config.pretrained_checkpoint)
      # We only want the embedding weights of the checkpoint we just restored.
      # So we re-initialize everything that's not an embedding weight. Also,
      # since this episodic finetuning procedure is a different optimization
      # problem than the original training of the baseline whose embedding
      # weights are re-used, we do not reload ADAM's variables and instead learn
      # them from scratch.
      vars_to_reinit, embedding_var_names, vars_to_reinit_names = [], [], []
      for var in tf.global_variables():
        if (any(keyword in var.name for keyword in EMBEDDING_KEYWORDS) and
            'adam' not in var.name.lower()):
          embedding_var_names.append(var.name)
          continue
        vars_to_reinit.append(var)
        vars_to_reinit_names.append(var.name)
      tf.logging.info('Initializing all variables except for %s.' %
                      embedding_var_names)
      self.sess.run(tf.variables_initializer(vars_to_reinit))
      tf.logging.info('Re-initialized vars %s.' % vars_to_reinit_names)

  def _create_held_out_specification(self, split='test'):
    """Create an EpisodeSpecification for either validation or testing.

    Note that testing is done episodically whether or not training was episodic.
    This is why the different subclasses should not override this method.

    Args:
      split: one of 'valid' or 'test'

    Returns:
      an EpisodeSpecification.

    Raises:
      ValueError: Invalid split.
    """
    split_enum = get_split_enum(split)
    return learning_spec.EpisodeSpecification(split_enum, self.num_test_classes,
                                              self.num_support_eval,
                                              self.num_query_eval)

  def _create_train_specification(self):
    """Returns an EpisodeSpecification or BatchSpecification for training.

    Raises:
      NotImplementedError: Should be implemented in each subclass.
    """
    raise NotImplementedError('Abstract Method.')

  def build_data(self, split):
    raise NotImplementedError('Abstract method.')

  def _restrict_dataset_list_for_split(self, split, splits_to_contribute,
                                       dataset_list):
    """Returns the restricted dataset_list for the given split.

    Args:
      split: A string, either 'train', 'valid' or 'test'.
      splits_to_contribute: A list whose length is the number of datasets in the
        benchmark. Each element is a set of strings corresponding to the splits
        that the respective dataset will contribute to.
      dataset_list: A list which has one element per selected dataset (same
        length as splits_to_contribute), e.g. this can be one of the lists
        dataset_spec_list, has_dag_ontology, has_bilevel_ontology of the
        BenchmarkSpecification.
    """
    updated_list = []
    for dataset_num, dataset_splits in enumerate(splits_to_contribute):
      if split in dataset_splits:
        updated_list.append(dataset_list[dataset_num])
    return updated_list

  def get_num_to_take(self, dataset_name, split):
    """Return the number of examples to restrict to for a dataset/split pair."""
    num_to_take = -1  # By default, no restriction.
    if dataset_name in self.restrict_num_per_class:
      dataset_restrict_num_per_class = self.restrict_num_per_class[dataset_name]
      if split in dataset_restrict_num_per_class:
        num_to_take = dataset_restrict_num_per_class[split]
    return num_to_take

  def build_episode(self, split):
    """Builds an EpisodeDataset containing the next data for "split".

    Args:
      split: A string, either 'train', 'valid', or 'test'.

    Returns:
      An EpisodeDataset.
    """
    shuffle_buffer_size = self.data_config.shuffle_buffer_size
    read_buffer_size_bytes = self.data_config.read_buffer_size_bytes
    num_prefetch = self.data_config.num_prefetch
    (_, image_shape, dataset_spec_list, has_dag_ontology, has_bilevel_ontology,
     splits_to_contribute) = self.benchmark_spec

    # Choose only the datasets that are chosen to contribute to the given split.
    dataset_spec_list = self._restrict_dataset_list_for_split(
        split, splits_to_contribute, dataset_spec_list)
    has_dag_ontology = self._restrict_dataset_list_for_split(
        split, splits_to_contribute, has_dag_ontology)
    has_bilevel_ontology = self._restrict_dataset_list_for_split(
        split, splits_to_contribute, has_bilevel_ontology)

    episode_spec = self.split_episode_or_batch_specs[split]
    dataset_split = episode_spec[0]
    # TODO(lamblinp): Support non-square shapes if necessary. For now, all
    # images are resized to square, even if it changes the aspect ratio.
    image_size = image_shape[0]
    if image_shape[1] != image_size:
      raise ValueError(
          'Expected a square image shape, not {}'.format(image_shape))

    if split == 'train':
      episode_descr_config = self.train_episode_config
    elif split == 'valid' or split == 'test':
      episode_descr_config = self.eval_episode_config
    else:
      raise ValueError('Unexpected split: {}. Was expecting "train", "valid" '
                       'or "test".'.format(split))

    # Decide how many examples per class to restrict to for each dataset for the
    # given split (by default there is no restriction).
    num_per_class = []  # A list whose length is the number of datasets.
    for dataset_spec in dataset_spec_list:
      num_per_class.append(self.get_num_to_take(dataset_spec.name, split))

    # TODO(lamblinp): pass specs directly to the pipeline builder.
    # TODO(lamblinp): move the special case directly in make_..._pipeline
    if len(dataset_spec_list) == 1:

      use_dag_ontology = has_dag_ontology[0]
      if self.eval_finegrainedness or self.eval_imbalance_dataset:
        use_dag_ontology = False
      data_pipeline = pipeline.make_one_source_episode_pipeline(
          dataset_spec_list[0],
          use_dag_ontology=use_dag_ontology,
          use_bilevel_ontology=has_bilevel_ontology[0],
          split=dataset_split,
          episode_descr_config=episode_descr_config,
          shuffle_buffer_size=shuffle_buffer_size,
          read_buffer_size_bytes=read_buffer_size_bytes,
          num_prefetch=num_prefetch,
          image_size=image_size,
          num_to_take=num_per_class[0])
    else:
      data_pipeline = pipeline.make_multisource_episode_pipeline(
          dataset_spec_list,
          use_dag_ontology_list=has_dag_ontology,
          use_bilevel_ontology_list=has_bilevel_ontology,
          split=dataset_split,
          episode_descr_config=episode_descr_config,
          shuffle_buffer_size=shuffle_buffer_size,
          read_buffer_size_bytes=read_buffer_size_bytes,
          num_prefetch=num_prefetch,
          image_size=image_size,
          num_to_take=num_per_class)

    data_pipeline = apply_dataset_options(data_pipeline)

    iterator = data_pipeline.make_one_shot_iterator()
    (support_images, support_labels, support_class_ids, query_images,
     query_labels, query_class_ids) = iterator.get_next()
    return providers.EpisodeDataset(
        train_images=support_images,
        test_images=query_images,
        train_labels=support_labels,
        test_labels=query_labels,
        train_class_ids=support_class_ids,
        test_class_ids=query_class_ids)

  def build_batch(self, split):
    """Builds a Batch object containing the next data for "split".

    Args:
      split: A string, either 'train', 'valid', or 'test'.

    Returns:
      An EpisodeDataset.
    """
    shuffle_buffer_size = self.data_config.shuffle_buffer_size
    read_buffer_size_bytes = self.data_config.read_buffer_size_bytes
    num_prefetch = self.data_config.num_prefetch
    (_, image_shape, dataset_spec_list, _, _,
     splits_to_contribute) = self.benchmark_spec

    # Choose only the datasets that are chosen to contribute to the given split.
    dataset_spec_list = self._restrict_dataset_list_for_split(
        split, splits_to_contribute, dataset_spec_list)

    # Decide how many examples per class to restrict to for each dataset for the
    # given split (by default there is no restriction).
    num_per_class = []  # A list whose length is the number of datasets.
    for dataset_spec in dataset_spec_list:
      num_per_class.append(self.get_num_to_take(dataset_spec.name, split))

    dataset_split, batch_size = self.split_episode_or_batch_specs[split]
    for dataset_spec in dataset_spec_list:
      if dataset_spec.name in DATASETS_WITH_EXAMPLE_SPLITS:
        raise ValueError(
            'Batch pipeline is used only at meta-train time, and does not '
            'handle datasets with example splits, which should only be used '
            'at meta-test (evaluation) time.')
    # TODO(lamblinp): pass specs directly to the pipeline builder.
    # TODO(lamblinp): move the special case directly in make_..._pipeline
    if len(dataset_spec_list) == 1:
      data_pipeline = pipeline.make_one_source_batch_pipeline(
          dataset_spec_list[0],
          split=dataset_split,
          batch_size=batch_size,
          shuffle_buffer_size=shuffle_buffer_size,
          read_buffer_size_bytes=read_buffer_size_bytes,
          num_prefetch=num_prefetch,
          image_size=image_shape[0],
          num_to_take=num_per_class[0])
    else:
      data_pipeline = pipeline.make_multisource_batch_pipeline(
          dataset_spec_list,
          split=dataset_split,
          batch_size=batch_size,
          shuffle_buffer_size=shuffle_buffer_size,
          read_buffer_size_bytes=read_buffer_size_bytes,
          num_prefetch=num_prefetch,
          image_size=image_shape[0],
          num_to_take=num_per_class)

    data_pipeline = apply_dataset_options(data_pipeline)
    iterator = data_pipeline.make_one_shot_iterator()
    images, class_ids = iterator.get_next()
    return providers.Batch(images=images, labels=class_ids)

  def get_next(self, split):
    """Returns the next batch or episode.

    Args:
      split: A str, one of 'train', 'valid', or 'test'.

    Raises:
      ValueError: Invalid split.
    """
    if split not in ['train', 'valid', 'test']:
      raise ValueError('Invalid split. Expected one of "train", "valid", or '
                       '"test".')
    return self.next_data[split]

  def get_train_op(self, global_step):
    """Returns the operation that performs a training update."""
    # UPDATE_OPS picks up batch_norm updates.
    update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
    with tf.control_dependencies(update_ops):
      train_op = self.optimizer.minimize(
          self.losses['train'], global_step=global_step)
    return train_op

  def get_updated_global_step(self):
    with tf.control_dependencies([self.train_op]):
      global_step = tf.identity(tf.train.get_global_step())
    return global_step

  def train(self):
    """The training loop."""
    global_step = self.sess.run(tf.train.get_global_step())
    tf.logging.info('Starting training from global_step: %d', global_step)
    updated_global_step = self.get_updated_global_step()

    # Dummy variables so that logging works even if called before evaluation.
    self.valid_acc = np.nan
    self.valid_ci = np.nan
    self.valid_other_metrics = {}
    self.valid_other_metrics_ci = {}

    # Compute the initial validation performance before starting the training.
    self.maybe_evaluate(global_step)

    while global_step < self.learn_config.num_updates:
      # Perform the next update.
      (_, train_loss, train_acc, train_metrics, train_tensors_for_metrics, global_step) = self.sess.run([
          self.train_op, self.losses['train'], self.accs['train'], self.metrics['train'],
          self.tensors_for_metrics['train'], updated_global_step
      ])

      # Only compute Clustering accuracy during evaluation
      #train_other_metrics = self.learners['train'].get_other_metrics(train_tensors_for_metrics)
      #tf.logging.info('Other metrics {}'.format(train_other_metrics))

      # Maybe validate, depending on the global step's value.
      self.maybe_evaluate(global_step)

      # Log training progress.
      if not global_step % self.learn_config.log_every:
        tf.logging.info('*'*32 + '\n')
        message = (
            'Update %d. Train loss: %f, Train accuracy: %f, '
            'Valid accuracy %f +/- %f.' %
            (global_step, train_loss, train_acc, self.valid_acc, self.valid_ci))
        tf.logging.info(message)

        # also print other metrics
        for key in train_metrics:
          tf.logging.info('-> Train metric {}: {:.5f}'.format(key, train_metrics[key]))

        for key in self.valid_other_metrics:
          tf.logging.info('-> Valid {}: {:.5f} +/- {:.5f}'.format(key, self.valid_other_metrics[key],
                                                                  self.valid_other_metrics_ci[key]))
        # Update summaries.
        summaries = self.sess.run(self.standard_summaries)
        if self.summary_writer:
          self.summary_writer.add_summary(summaries, global_step)

      should_save = self.checkpoint_dir is not None
      if should_save and global_step % self.learn_config.checkpoint_every == 0:
        save_path = self.saver.save(
            self.sess,
            os.path.join(self.checkpoint_dir, 'model_%d.ckpt' % global_step))
        tf.logging.info('Model checkpoint saved: %s' % save_path)

  def maybe_evaluate(self, global_step):
    """Maybe perform evaluation, depending on the value of global_step."""
    if not global_step % self.learn_config.validate_every:
      # Get the validation accuracy and confidence interval.
      (valid_acc, valid_ci,
       valid_acc_summary, valid_ci_summary,
       valid_other_metrics, valid_other_metrics_ci,
       valid_other_metrics_summary, valid_other_metrics_ci_summary) = self.evaluate('valid')
      # Validation summaries are updated every time validation happens which is
      # every validate_every steps instead of log_every steps.
      # Those are called mean valid acc / test valid ci
      # as opposed to the running averages
      if self.summary_writer:
        self.summary_writer.add_summary(valid_acc_summary, global_step)
        self.summary_writer.add_summary(valid_ci_summary, global_step)

        # Also write evaluation metrics
        for key in valid_other_metrics_summary:
          self.summary_writer.add_summary(valid_other_metrics_summary[key], global_step)
          self.summary_writer.add_summary(valid_other_metrics_ci_summary[key], global_step)

      self.valid_acc = valid_acc
      self.valid_ci = valid_ci

      # Save for logging during training
      self.valid_other_metrics = valid_other_metrics
      self.valid_other_metrics_ci = valid_other_metrics_ci


  def evaluate(self, split):
    """Returns performance metrics across num_eval_trials episodes / batches."""

    # Constants
    UA = 'UnsupervisedAcc_softmax'
    SA = 'SupervisedAcc'

    num_eval_trials = self.learn_config.num_eval_episodes
    tf.logging.info(
        'Performing evaluation of the %s split using %d episodes...' %
        (split, num_eval_trials))
    accuracies = []
    all_other_metrics = {}

    for eval_trial_num in tqdm(range(num_eval_trials)):
      acc, metrics, tensors_for_metrics, summaries = self.sess.run(
          [self.accs[split], self.metrics[split], self.tensors_for_metrics[split], self.evaluation_summaries])
      accuracies.append(acc)

      # Compute metrics
      # Limit number of times unless it is test split
      other_metrics = {}
      if split == 'test' or self.learn_config.num_eval_other_metrics==0 or eval_trial_num < self.learn_config.num_eval_other_metrics:
        other_metrics = self.learners[split].get_other_metrics(tensors_for_metrics)

      # Accumulate
      for key in other_metrics:
        all_other_metrics.setdefault(key, [])
        all_other_metrics[key].append(other_metrics[key])
      # Redundant, accumulate accuracy too
      all_other_metrics.setdefault(SA, [])
      all_other_metrics[SA].append(acc)

      # Log output
      if eval_trial_num % 30 == 0 and split=='test':
        # Reduce first
        mean_other_metrics = {key:np.mean(val) for key,val in all_other_metrics.items()}
        ci_other_metrics = {key:1.96 * np.std(val) / np.sqrt(len(val)) for key,val in all_other_metrics.items()}
        tf.logging.info('Eval trial {}/{}'.format(eval_trial_num, num_eval_trials))
        for key in mean_other_metrics:
          tf.logging.info('-> {} {}: {:.5f} +/- {:.5f}'.format(split, key, mean_other_metrics[key], ci_other_metrics[key]))

      # Write evaluation summaries.
      if split == self.eval_split and self.summary_writer:
        self.summary_writer.add_summary(summaries, eval_trial_num)

      # If split is test, it's nice to track progress as we go
      # In that case, evaluate() is run only once.
      if split=='test' and self.summary_writer:
        self.summary_writer.add_summary(make_summary('EVAL_test_acc', acc), eval_trial_num)
        for key in other_metrics:
          self.summary_writer.add_summary(make_summary('EVAL_test_{}'.format(key), other_metrics[key]), eval_trial_num)


    tf.logging.info('Evaluation Done.')

    # Accumulate accuracy
    mean_acc = np.mean(accuracies)
    ci_acc = np.std(accuracies) * 1.96 / np.sqrt(len(accuracies))  # confidence
    # Generate summaries for maybe_evaluate()
    mean_acc_summary = tf.Summary()
    mean_acc_summary.value.add(tag='mean %s acc' % split, simple_value=mean_acc)
    ci_acc_summary = tf.Summary()
    ci_acc_summary.value.add(tag='%s acc CI' % split, simple_value=ci_acc)

    # Reduce other metrics
    mean_other_metrics = {key:np.mean(val) for key,val in all_other_metrics.items()}
    ci_other_metrics = {key:1.96 * np.std(val) / np.sqrt(len(val)) for key,val in all_other_metrics.items()}

    # Introduce a compound metric , CSCC
    if UA in mean_other_metrics and SA in mean_other_metrics:
      cscc_mean, cscc_ci = get_uncertain_ratio(mean_other_metrics[UA], mean_other_metrics[SA],
              ci_other_metrics[UA], ci_other_metrics[SA])
      print ('UA',mean_other_metrics[UA])
      print ('SA',mean_other_metrics[SA])
      mean_other_metrics['CSCC'] = cscc_mean
      ci_other_metrics['CSCC'] = cscc_ci

    # Generate summaries for maybe_evaluate()
    mean_other_metrics_summary = {}
    ci_other_metrics_summary = {}
    for key in mean_other_metrics:
      mean_other_metrics_summary[key] = make_summary('{}_mean_{}'.format(split, key), mean_other_metrics[key])
      ci_other_metrics_summary[key] = make_summary('{}_ci_{}'.format(split, key), ci_other_metrics[key])

    if split == 'test':
      tf.logging.info('Test accuracy: %f, +/- %f.\n' % (mean_acc, ci_acc))
      # Text Summary
      lines = [
        'Dataset: {} ({} episodes)'.format('|'.join(self.eval_dataset_list), num_eval_trials),
        'Test accuracy: {:.5f}, +/- {:.5f}'.format(mean_acc, ci_acc)
      ]
      for key in mean_other_metrics:
        lines.append('Test {}: {:.5f}, +/- {:.5f}'.format(key, mean_other_metrics[key], ci_other_metrics[key]))
      self.summary_writer.add_summary(make_text_summary('Summary', '<br>\n'.join(lines)), num_eval_trials)
      tf.logging.info('*'*32)
      tf.logging.info('\n'.join(lines))
      tf.logging.info('*'*32)

      # Write the lines to file
      if not os.path.exists(self.summary_dir):
        os.makedirs(self.summary_dir)
      test_summary_path = os.path.join(self.summary_dir, 'test_summary.txt')
      with open(test_summary_path, 'w') as fp:
        fp.write('\n'.join(lines))
      print ('Wrote summary to ', test_summary_path)

    return (mean_acc, ci_acc, mean_acc_summary, ci_acc_summary,
            mean_other_metrics, ci_other_metrics, mean_other_metrics_summary, ci_other_metrics_summary)


  def add_eval_summaries_split(self, split):
    """Returns split's summaries of way / shot / classes / logits / targets."""
    split_eval_summaries = []
    way_summary = tf.summary.scalar('%s_way' % split, self.way[split])
    shots_summary = tf.summary.tensor_summary('%s_shots' % split,
                                              self.shots[split])
    classes_summary = tf.summary.tensor_summary('%s_class_ids' % split,
                                                self.class_ids[split])
    logits_summary = tf.summary.tensor_summary('%s_test_logits' % split,
                                               self.test_logits[split])
    targets_summary = tf.summary.tensor_summary('%s_test_targets' % split,
                                                self.test_targets[split])
    if self.eval_imbalance_dataset:
      class_props_summary = tf.summary.tensor_summary('%s_class_props' % split,
                                                      self.class_props[split])
      split_eval_summaries.append(class_props_summary)
    split_eval_summaries.append(way_summary)
    split_eval_summaries.append(shots_summary)
    split_eval_summaries.append(classes_summary)
    split_eval_summaries.append(logits_summary)
    split_eval_summaries.append(targets_summary)
    return split_eval_summaries


class EpisodicTrainer(Trainer):
  """A Trainer that trains a learner through a series of episodes."""

  def build_data(self, split):
    # An EpisodicTrainer will use episodes for all splits.
    return self.build_episode(split)

  def _create_train_specification(self):
    """Returns an EpisodeSpecification or BatchSpecification for training."""
    return learning_spec.EpisodeSpecification(learning_spec.Split.TRAIN,
                                              self.num_train_classes,
                                              self.num_support_train,
                                              self.num_query_train)

  def set_way_shots_classes_logits_targets(self):
    """Sets the Tensors for the above info of the learner's next episode."""
    self.maybe_set_way_shots_classes_logits_targets()

  def add_eval_summaries(self):
    """Returns summaries of way / shot / classes/ logits / targets."""
    evaluation_summaries = []
    for split in self.required_splits:
      evaluation_summaries.extend(self.add_eval_summaries_split(split))
    return evaluation_summaries

  def create_train_learner(self, train_learner_class, episode_or_batch):
    """Instantiates a train learner."""
    return train_learner_class(True, self.learn_config.transductive_batch_norm,
                               self.backprop_through_moments, self.ema_object,
                               self.embedding_fn, episode_or_batch)

  def create_eval_learner(self, eval_learner_class, episode):
    """Instantiates an eval learner."""
    return eval_learner_class(False, self.learn_config.transductive_batch_norm,
                              self.backprop_through_moments, self.ema_object,
                              self.embedding_fn, episode)


class BatchTrainer(Trainer):
  """A Trainer that trains a learner through a series of batches."""

  def build_data(self, split):
    # The 'train' split is read as batches, 'valid' and 'test' as episodes.
    if split == 'train' and self.eval_split != 'train':
      return self.build_batch(split)
    elif split in ('train', 'valid', 'test'):
      return self.build_episode(split)
    else:
      raise ValueError('Unexpected value for split: {}'.format(split))

  def _get_num_total_classes(self):
    """Returns total number of classes in the benchmark."""
    # Total class is needed because for ImageNet instead of linearly assigning
    # class id's we use the hierarchy to do this, thus the assumption:
    # {train_ids} < {val_ids} < {test_ids} doesn't hold.
    total_classes = 0
    for dataset_spec in self.benchmark_spec.dataset_spec_list:
      for split in learning_spec.Split:
        total_classes += len(dataset_spec.get_classes(split))
    return total_classes

  def _create_train_specification(self):
    """Returns an EpisodeSpecification or BatchSpecification for training."""
    if self.eval_split == 'train':
      return learning_spec.EpisodeSpecification(learning_spec.Split.TRAIN,
                                                self.num_train_classes,
                                                self.num_support_train,
                                                self.num_query_train)
    else:
      return learning_spec.BatchSpecification(learning_spec.Split.TRAIN,
                                              self.learn_config.batch_size)

  def set_way_shots_classes_logits_targets(self):
    """Sets the Tensors for the above info of the learner's next episode."""
    skip_train = True
    if self.eval_split == 'train':
      skip_train = False
    self.maybe_set_way_shots_classes_logits_targets(skip_train=skip_train)

  def add_eval_summaries(self):
    """Returns summaries of way / shot / class id's / logits / targets."""
    evaluation_summaries = []
    for split in self.required_splits:
      # In the Batch case, training is non-episodic but evaluation is episodic.
      if split == 'train' and self.eval_split != 'train':
        continue
      evaluation_summaries.extend(self.add_eval_summaries_split(split))
    return evaluation_summaries

  def create_train_learner(self, train_learner_class, episode_or_batch):
    """Instantiates a train learner."""
    num_total_classes = self._get_num_total_classes()
    is_training = False if self.eval_split == 'train' else True
    return train_learner_class(is_training,
                               self.learn_config.transductive_batch_norm,
                               self.backprop_through_moments, self.ema_object,
                               self.embedding_fn, episode_or_batch,
                               num_total_classes, self.num_test_classes)

  def create_eval_learner(self, eval_learner_class, episode):
    """Instantiates an eval learner."""
    num_total_classes = self._get_num_total_classes()
    return eval_learner_class(False, self.learn_config.transductive_batch_norm,
                              self.backprop_through_moments, self.ema_object,
                              self.embedding_fn, episode, num_total_classes,
                              self.num_test_classes)
