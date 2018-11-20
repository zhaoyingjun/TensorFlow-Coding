import tensorflow as tf

import getConfig

gConfig={}

gConfig=getConfig.get_config(config_file='config.ini')


class dnnClassifierModel(object):

	def __init__(self):


		def createModel(feature_columns):
			return tf.contrib.learn.DNNClassifier(
				feature_columns=feature_cols,
				hidden_units=[105, 105, 105, 105 ],
				model_dir=gConfig['model_dir'])
		