import tensorflow as tf #version=1.4 , 최신 버전 쓰면 beamsearch + attention인 경우에 alginemnt_history = True로 할 수 있음. 1.4버전은 안됨.
import numpy as np
import csv 
import os

train_set_path = './dataset/train_set.csv'
vali_set_path = './dataset/vali_set.csv'
test_set_path = './dataset/test_set.csv'
saver_path = './saver_attention/'
tensorboard_path = './tensorboard_attention'

maximum_encoder_length = 11 # 5digit + 1operator + 5digit
maximum_decoder_length = 8  # 'go' + 6digit + 'eos'  => 나중에는 'go' + 6digit = decoder input,  6digit + 'eos' = decoder target 로 분리해서 씀.
maximum_target_length = maximum_decoder_length-1 #7 == 6digit+'eos' 

#de_pad는 embedding 값 00 아님. 만약 00이면 lstm값도 전부 0이라서 softmax 불가능,   #en_pad => embedding 값 00됨.
dic = {'0':0, '1':1, '2':2, '3':3, '4':4, '5':5, '6':6, '7':7, '8':8, 
		'9':9, '+':10, '-':11, 'go':12, 'eos':13, 'de_pad':14, 'en_pad':15}


# 1차원 리스트가 인풋임. read_csv에서 사용.
#expression 부분은 11자리가 아니면 앞에서부터 0으로 패딩, 인코더 인풋으로 사용. 
#result 부분은 디코더 인풋, 최대 8자리가 되도록 패딩.
def data_preprocess(expression, result, mode='train'): 
	if mode == 'train':
		#https://docs.scipy.org/doc/numpy/reference/generated/numpy.pad.html
		expression = np.pad(expression, (maximum_encoder_length - len(expression), 0), 'constant', constant_values = dic['en_pad'])

		#최대 8자리가 되도록 패딩, result앞에 go, result 뒤에 eos, eos 뒤에 0
		decoder_size = len(result)+1 # 실제 디코더에서는 진짜 result+eos만 사용되는데, 텐서플로우에서 seq2seq 학습 시에는 이 길이가 필요함.
		result = np.pad(result, (1, 0), 'constant', constant_values = dic['go']) # 앞 1자리, 뒤 0자리에 'go' 넣어라.
		result = np.pad(result, (0, 1), 'constant', constant_values = dic['eos'])
		result = np.pad(result, (0, maximum_decoder_length - len(result)), 'constant', constant_values = dic['de_pad']) 

		return np.append( np.append(expression, result), decoder_size ) # 나중에 maximum_encoder_length, maximum_decoder_length, 1 => 3개로 분리해서 쓰면 됨

	else: # test
		expression = np.pad(expression, (maximum_encoder_length - len(expression), 0), 'constant', constant_values = dic['en_pad'])
		
		#최대 7자리가 되도록 패딩, result 뒤에 eos, eos 뒤에 0
		result = np.pad(result, (0, 1), 'constant', constant_values = dic['eos'])
		result = np.pad(result, (0, maximum_target_length - len(result)), 'constant', constant_values = -1) #정확도 체크용

		return np.append(expression, result)



def read_csv(path, mode='train'):
	data = []
	with open(path, 'r', newline='') as o:
		wr = csv.reader(o)
		for index, i in enumerate(wr):

			#if index == 2000:
			#	break
			
			row = ''.join(i).split('=') # "226+138=364" => ['226+138', '364']
			expression = list(row[0])  #"226+138"
			result = list(row[1]) # '364'

			for k in range(len(expression)):
				expression[k] = dic[expression[k]]  # 1차원 list 형태 ex [2, 1, 4, 1, 8, 10, 2, 2, 0, 3]
			for k in range(len(result)): 
				result[k] = dic[result[k]] # 1차원 list 형태 ex [2, 3, 6, 2, 1]
			
			preprocessed = data_preprocess(expression, result, mode=mode) # get_batch에서 11, 8, 1로 끊어 읽으면 됨
			data.append(preprocessed)

	return np.array(data)



def get_batch(data, mode = 'train'):
	if mode == 'train':
		en_input = data[:, :maximum_encoder_length]
		de_input = data[:, maximum_encoder_length:maximum_encoder_length+maximum_decoder_length]
		de_sequence_length = data[:,-1]
		return en_input, de_input, de_sequence_length

	else:
		en_input = data[:, :maximum_encoder_length]
		de_input = data[:, maximum_encoder_length:]
		return en_input, de_input



def train(data, model):
	train_batch_size = 256
	loss = 0
	np.random.shuffle(data)
	
	for i in range( int(np.ceil(len(data)/train_batch_size)) ):
		#print("batch:", i+1, '/', int(np.ceil(len(data)/train_batch_size)) )
		en_input, de_input, de_sequence_length = get_batch(data[train_batch_size*i:train_batch_size*(i+1)])

		_, train_loss = sess.run([model.minimize, model.cost], {
						model.x:en_input, 
						model.decoder_input:de_input[:,:-1], 
						model.decoder_target:de_input[:, 1:], 
						model.target_sequence_length:de_sequence_length,
						model.batch_size:len(en_input)
					}
				)
		loss += train_loss

	return loss



def validation(data, model):
	vali_batch_size = 256
	loss = 0

	for i in range( int(np.ceil(len(data)/vali_batch_size)) ):
		en_input, de_input, de_sequence_length = get_batch(data[vali_batch_size*i:vali_batch_size*(i+1)])
		vali_loss = sess.run(model.cost, {
						model.x:en_input, 
						model.decoder_input:de_input[:, :-1], 
						model.decoder_target:de_input[:, 1:], 
						model.target_sequence_length:de_sequence_length,
						model.batch_size:len(en_input)
					}
				)
		loss += vali_loss

	return loss



def test(data, model): #beam 1~3 test
	test_batch_size = 1024
	correct = 0 #beam1
	
	for i in range( int(np.ceil(len(data)/test_batch_size)) ):
		en_input, target = get_batch(data[test_batch_size*i:test_batch_size*(i+1)], mode='test')
		result = sess.run(model.best_beam_output, {
						model.x:en_input, 
						model.batch_size:len(en_input)
					}
				)				
		correct += np.sum(np.all(np.equal(result, target), axis=1))
	
	return correct / len(data)



def run(train_set, vali_set, test_set, model, restore=-1):
	#weight save path
	if not os.path.exists(saver_path):
		os.makedirs(saver_path)

	#restore check
	if restore != -1:
		model.saver.restore(sess, saver_path+str(restore)+".ckpt")
	else:
		restore = 0

	#train, vali, test
	for epoch in range(restore + 1, 300):
		train_loss = train(train_set, model)
		vali_loss = validation(vali_set, model)
		accuracy = test(test_set, model)

		print("epoch : ", epoch, " train_loss : ", train_loss, " vali_loss : ", vali_loss, " accuracy : ", accuracy)

		summary = sess.run(model.merged, {
							model.train_loss_tensorboard:train_loss, 
							model.vali_loss_tensorboard:vali_loss,
							model.test_accuracy_tensorboard:accuracy, 
						}
					)
		model.writer.add_summary(summary, epoch)
				
		if epoch % 5 == 0:
			print('\n')
			save_path = model.saver.save(sess, saver_path+str(epoch)+".ckpt")

		

class seq2seq_attention:
	def __init__(self, sess):

		self.attention_mechanism = "Luong" #or "Bahdanau"

		####hyper parameter
		self.learning_rate = 0.0005
		self.cell_num = 256
		self.decoder_cell_num = self.cell_num*2 #bidirectional LSTM
		self.target_size = 15 # 0 ~ 14 
		self.embedding_dimension = 2
		self.beam_width = 1
		self.alignment_history = False # beam+attention 할때는 false 해야됨. 최신버전쓰면 true해도 됨.


		#Model
		self.target_sequence_length = tf.placeholder(tf.int32, [None], name = "target_sequence_length") 
		self.batch_size = tf.placeholder(tf.int32, [], name = 'batch_size')

		self.x = tf.placeholder(tf.int32, [None, None], name = 'x') #batchsize, input_length
		self.decoder_input = tf.placeholder(tf.int32, [None, None], name = 'decoder_input') #decoder input => 'go', sequence
		self.decoder_target = tf.placeholder(tf.int32, [None, None], name = 'decoder_target') # decoder_target => sequence, 'eos'

		self.embedding = tf.Variable(tf.random_uniform([self.target_size, self.embedding_dimension], -1., 1.)) #if embedding lookup size is bigger than embedding . then output is 0
		self.embedding_encoder = tf.nn.embedding_lookup(self.embedding, self.x) #target_size가 14이므로 0~13의 값만 갖는데 만약 x가 14 이상이면 00..0..00 리턴
		self.embedding_decoder = tf.nn.embedding_lookup(self.embedding, self.decoder_input)
		

		# encoder
		self.tiled_encoder_output, self.tiled_encoder_final_state = self.encoder() 


		# decoder
		self.output_layer = tf.layers.Dense(self.target_size) #output_layer, decoder_cell must write here. not in def function
		self.decoder_cell = tf.nn.rnn_cell.LSTMCell(self.decoder_cell_num) # encoder is bidirectional, so decoder_cell_num is 2*encoder_cell_num
		
		self.attention_cell, self.initial_state = self.Luong_attention(self.decoder_cell, self.tiled_encoder_output, self.tiled_encoder_final_state) # Luong_attention, Bahdanau_attention

		_, self.logits, self.output = self.train_decoder(self.attention_cell, self.initial_state, self.output_layer)	 
		_, self.best_beam_output = self.beam_decoder(self.attention_cell, self.initial_state, self.output_layer) #beam_decoder, greedy_decoder



		####weight update,  https://www.tensorflow.org/versions/r1.4/api_docs/python/tf/contrib/seq2seq/sequence_loss
		# masks = tf.sequence_mask([3,2,3], 5, dtype=tf.float32, name='masks') # [[1 1 1 0 0], [1.1.0.0.0], [1.1.1.0.0]]
		self.masks = tf.sequence_mask(self.target_sequence_length, maximum_target_length, dtype=tf.float32, name='masks') #sequence에서 의미 있는 길이만 사용하려고 mask 활용함.
		self.cost = tf.contrib.seq2seq.sequence_loss(self.logits, self.decoder_target, self.masks)
		self.optimizer = tf.train.AdamOptimizer(self.learning_rate)
		self.minimize = self.optimizer.minimize(self.cost)
		

		##tensorboard
		self.train_loss_tensorboard = tf.placeholder(tf.float32, name='train_loss')
		self.vali_loss_tensorboard = tf.placeholder(tf.float32, name='vali_loss')
		self.test_accuracy_tensorboard = tf.placeholder(tf.float32, name='test')

		self.train_summary = tf.summary.scalar("train_loss", self.train_loss_tensorboard)
		self.vali_summary = tf.summary.scalar("vali_loss", self.vali_loss_tensorboard)
		self.test_summary = tf.summary.scalar("test_accuracy", self.test_accuracy_tensorboard)
		
		self.merged = tf.summary.merge_all()
		self.writer = tf.summary.FileWriter(tensorboard_path, sess.graph)


		#variable initialize
		sess.run(tf.global_variables_initializer())
		
		#saver
		self.saver = tf.train.Saver(max_to_keep=10000)



	def encoder(self):#, fw, bw, beam = 1): # beam이 1이면 train 용으로 사용 가능. tile_batch 결과가 수행 전과 동일하므로.
		fw = tf.nn.rnn_cell.LSTMCell(self.cell_num) # cell must write here. not in def function
		bw = tf.nn.rnn_cell.LSTMCell(self.cell_num)
		#tiled_encoder_output, tiled_encoder_final_state = encoder(fw, bw, beam_width)

		((en_fw_val, en_bw_val), (en_fw_state, en_bw_state)) = tf.nn.bidirectional_dynamic_rnn(fw, bw, self.embedding_encoder, dtype=tf.float32)
		en_val_concat = tf.concat((en_fw_val, en_bw_val), 2) 
		en_state_c = tf.concat((en_fw_state.c, en_bw_state.c), 1)
		en_state_h = tf.concat((en_fw_state.h, en_bw_state.h), 1)
		en_state_concat = tf.contrib.rnn.LSTMStateTuple(c = en_state_c, h = en_state_h)

		#for attention, beam search.
		tiled_encoder_output = tf.contrib.seq2seq.tile_batch(en_val_concat, self.beam_width)
		tiled_encoder_final_state = tf.contrib.seq2seq.tile_batch(en_state_concat, self.beam_width)

		return tiled_encoder_output, tiled_encoder_final_state



	def Luong_attention(self, decoder_cell, tiled_encoder_output, tiled_encoder_final_state):
		LuongAttention = tf.contrib.seq2seq.LuongAttention(
				num_units = self.decoder_cell_num, 
				memory = tiled_encoder_output, 
				name = "Luong"
			)
		attention_cell = tf.contrib.seq2seq.AttentionWrapper(
				cell = decoder_cell, 
				attention_mechanism = LuongAttention, 
				attention_layer_size = self.decoder_cell_num, 
				alignment_history = self.alignment_history #이게 true면 attention+beam_search 동작 안함.... false로 해야함. https://github.com/tensorflow/tensorflow/pull/13312
			)
		initial_state = attention_cell.zero_state(self.batch_size * self.beam_width, tf.float32).clone(cell_state=tiled_encoder_final_state) 

		return attention_cell, initial_state


	
	def Bahdanau_attention(self, decoder_cell, tiled_encoder_output, tiled_encoder_final_state):
		BahdanauAttention = tf.contrib.seq2seq.BahdanauAttention(
				num_units = self.decoder_cell_num, 
				memory = tiled_encoder_output, 
				name = "Bahdanau"
			)
		attention_cell = tf.contrib.seq2seq.AttentionWrapper(
				cell = decoder_cell, 
				attention_mechanism = BahdanauAttention, 
				attention_layer_size = self.decoder_cell_num, 
				alignment_history = self.alignment_history #이게 true면 attention+beam_search 동작 안함.... false로 해야함.
			)
		initial_state = attention_cell.zero_state(self.batch_size * self.beam_width, tf.float32).clone(cell_state=tiled_encoder_final_state) 

		return attention_cell, initial_state
	


	def train_decoder(self, attention_cell, initial_state, output_layer): #beam이 1이면 greedy search 임
		train_helper = tf.contrib.seq2seq.TrainingHelper(
				inputs = self.embedding_decoder, 
				sequence_length = self.target_sequence_length
			)
		
		train_decoder = tf.contrib.seq2seq.BasicDecoder(
				cell = attention_cell, 
				helper = train_helper, 
				initial_state = initial_state, 
				output_layer = output_layer
			)
		
		decoder_output, _, _ = tf.contrib.seq2seq.dynamic_decode(
				decoder = train_decoder, 
				output_time_major = False, 
				impute_finished = True, 
				maximum_iterations = maximum_target_length
			)
		
		return decoder_output, decoder_output.rnn_output, decoder_output.sample_id



	def beam_decoder(self, attention_cell, initial_state, output_layer):
		decoder = tf.contrib.seq2seq.BeamSearchDecoder(  # beamsearch
				cell = attention_cell, 
				embedding = self.embedding, 
				start_tokens = tf.tile([dic['go']], [self.batch_size]), 
				end_token = dic['eos'], 
				initial_state = initial_state,				
				beam_width = self.beam_width,  # 1이면 greedy. 
				output_layer = output_layer
			)
		
		decoder_output, _, _ = tf.contrib.seq2seq.dynamic_decode(
				decoder = decoder, 
				output_time_major = False, 
				impute_finished = False, #beam search의 경우 이 부분은 False
				maximum_iterations = maximum_target_length
			)	#decoder_output[0].shape = [batch, time, beam]
		
		transe_output = tf.transpose(decoder_output[0], [2,0,1]) # [beam, batch, time]
		best_output = transe_output[0] #빔 단위로 나뉘고, 그 안에서 batch 단위로 묶이니깐 0번째 쓰면 젤높은확률의 빔 + 배치별로 하나씩 다 묶임.

		return decoder_output, best_output



	def greedy_decoder(self, attention_cell, initial_state, output_layer):
		#greedy search is equal to beam search when beam_width == 1
		helper = tf.contrib.seq2seq.GreedyEmbeddingHelper(
				embedding = self.embedding, 
				start_tokens = tf.tile([dic['go']], [self.batch_size]),
				end_token = dic['eos']
			) 
		decoder = tf.contrib.seq2seq.BasicDecoder(
				cell = attention_cell, 
				helper = helper, 
				initial_state = initial_state, 
				output_layer = output_layer
			)
		decoder_output, _, _ = tf.contrib.seq2seq.dynamic_decode(  # sequence is decoder_output[1]
				decoder = decoder, 
				output_time_major = False, 
				impute_finished = True, 
				maximum_iterations = maximum_target_length
			)
		
		return decoder_output, decoder_output[1]



	def testcode_greedy_equals_beam_when_beamsize_is_1(self):
		test_set = read_csv(test_set_path, 'test')
		en_input, de_output = get_batch(test_set[0:10], 'test')

		print('input\n', en_input)
		print('target\n',de_output)
		
		_, greedy_output = self.greedy_decoder(self.attention_cell, self.initial_state, self.output_layer)
		_, beam_best_output = self.beam_decoder(self.attention_cell, self.initial_state, self.output_layer)

		result = sess.run(greedy_output, {self.x:en_input, self.batch_size:len(en_input)})
		print('greedy\n',result,'\n')

		result = sess.run(beam_best_output, {self.x:en_input, self.batch_size:len(en_input)})
		print('beam_with:1\n',result)




####run
train_set = read_csv(train_set_path)
vali_set = read_csv(vali_set_path)
test_set = read_csv(test_set_path, 'test')

sess = tf.Session()
calc = seq2seq_attention(sess)
run(train_set, vali_set, test_set, calc, restore=-1)


##test code
#calc.testcode_greedy_equals_beam_when_beamsize_is_1()
