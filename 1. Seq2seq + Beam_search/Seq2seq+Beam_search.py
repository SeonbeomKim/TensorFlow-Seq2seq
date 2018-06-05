import tensorflow as tf #version=1.4
import numpy as np
import csv 
import os

train_set_path = './dataset/train_set.csv'
vali_set_path = './dataset/vali_set.csv'
test_set_path = './dataset/test_set.csv'
saver_path = './saver_non_attention/'
tensorboard_path = './tensorboard_non_attention'

####hyper parameter
learning_rate = 0.0005
cell_num = 256
target_size = 15 # 0 ~ 14 
embedding_dimension = 2
maximum_encoder_length = 11 # 5digit + 1operator + 5digit
maximum_decoder_length = 8  # 'go' + 6digit + 'eos'  => 나중에는 'go' + 6digit = decoder input,  6digit + 'eos' = decoder target 로 분리해서 씀.
maximum_target_length = maximum_decoder_length-1 #7 == 6digit+'eos' 
beam_width = 1

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

			#if index == 20000:
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
		


def train(data):
	train_batch_size = 256
	loss = 0
	np.random.shuffle(data)
	
	for i in range( int(np.ceil(len(data)/train_batch_size)) ):
		#print("batch:", i+1, '/', int(np.ceil(len(data)/train_batch_size)) )
		en_input, de_input, de_sequence_length = get_batch(data[train_batch_size*i:train_batch_size*(i+1)])

		_, train_loss = sess.run([minimize, cost], 
				{x:en_input, decoder_input:de_input[:,:-1], decoder_target:de_input[:, 1:], target_sequence_length:de_sequence_length})
		loss += train_loss

	return loss
	


def validation(data):
	vali_batch_size = 256
	loss = 0

	for i in range( int(np.ceil(len(data)/vali_batch_size)) ):
		en_input, de_input, de_sequence_length = get_batch(data[vali_batch_size*i:vali_batch_size*(i+1)])
		vali_loss = sess.run(cost, 
				{x:en_input, decoder_input:de_input[:, :-1], decoder_target:de_input[:, 1:], target_sequence_length:de_sequence_length})
		loss += vali_loss

	return loss



def test(data): #beam 1~3 test
	test_batch_size = 1024
	correct1 = 0 #beam1
	correct2 = 0 #beam2
	correct3 = 0 #beam3

	for i in range( int(np.ceil(len(data)/test_batch_size)) ):
		en_input, target = get_batch(data[test_batch_size*i:test_batch_size*(i+1)], mode='test')
		result1 = sess.run(best_beam_1, {x:en_input, batch_size:len(en_input)})
		result2 = sess.run(best_beam_2, {x:en_input, batch_size:len(en_input)})
		result3 = sess.run(best_beam_3, {x:en_input, batch_size:len(en_input)})
				
		correct1 += np.sum(np.all(np.equal(result1, target), axis=1))
		correct2 += np.sum(np.all(np.equal(result2, target), axis=1))
		correct3 += np.sum(np.all(np.equal(result3, target), axis=1))
		

	return correct1 / len(data), correct2 / len(data), correct3 / len(data)



def run(train_set, vali_set, test_set, restore=-1):
	#weight save path
	if not os.path.exists(saver_path):
		os.makedirs(saver_path)

	#restore check
	if restore != -1:
		saver.restore(sess, saver_path+str(restore)+".ckpt")
	else:
		restore = 0

	#train, vali, test
	for epoch in range(restore + 1, 300):
		train_loss = train(train_set)
		vali_loss = validation(vali_set)
		print("epoch : ", epoch, " train_loss : ", train_loss, " vali_loss : ", vali_loss)
				
		if epoch % 5 == 0:
			save_path = saver.save(sess, saver_path+str(epoch)+".ckpt")

			accuracy1, accuracy2, accuracy3 = test(test_set)
			print("epoch : ", epoch, " accuracy1 : ", accuracy1, " accuracy2 : ", accuracy2, " accuracy3 : ", accuracy3, '\n')

			summary = sess.run(merged, {
							train_loss_tensorboard:train_loss, 
							vali_loss_tensorboard:vali_loss,
							test_accuracy_tensorboard1:accuracy1, 
							test_accuracy_tensorboard2:accuracy2, 
							test_accuracy_tensorboard3:accuracy3
						}
					)
			writer.add_summary(summary, epoch)
		


def encoder(fw, bw):
	((en_fw_val, en_bw_val), (en_fw_state, en_bw_state)) = tf.nn.bidirectional_dynamic_rnn(fw, bw, embedding_encoder, dtype=tf.float32)
	en_val_concat = tf.concat((en_fw_val, en_bw_val), 2) 
	en_state_c = tf.concat((en_fw_state.c, en_bw_state.c), 1)
	en_state_h = tf.concat((en_fw_state.h, en_bw_state.h), 1)
	en_state_concat = tf.contrib.rnn.LSTMStateTuple(c = en_state_c, h = en_state_h)

	return en_val_concat, en_state_concat



#https://www.tensorflow.org/versions/master/api_guides/python/contrib.seq2seq
#https://www.tensorflow.org/versions/r1.4/api_docs/python/tf/contrib/seq2seq/dynamic_decode
def train_decoder(output_layer, decoder_cell): #beam이 1이면 greedy search 임
	train_helper = tf.contrib.seq2seq.TrainingHelper(
			inputs = embedding_decoder, 
			sequence_length = target_sequence_length
		)
	
	train_decoder = tf.contrib.seq2seq.BasicDecoder(
			cell = decoder_cell, 
			helper=train_helper, 
			initial_state = en_state_concat, 
			output_layer = output_layer
		)
	
	decoder_output, _, _ = tf.contrib.seq2seq.dynamic_decode(
			decoder = train_decoder, 
			output_time_major = False, 
			impute_finished = True, 
			maximum_iterations = maximum_target_length
		)
	
	return decoder_output, decoder_output.rnn_output, decoder_output.sample_id



def beam_decoder(output_layer, decoder_cell, beam = 1):
	decoder = tf.contrib.seq2seq.BeamSearchDecoder(  # beamsearch
			cell = decoder_cell, 
			embedding = embedding, 
			start_tokens = tf.tile([dic['go']], [batch_size]), 
			end_token = dic['eos'], 
			initial_state = tf.contrib.seq2seq.tile_batch(en_state_concat, beam),				
			beam_width = beam,  # 1이면 greedy. 
			output_layer = output_layer
		)
	
	decoder_output, _, _ = tf.contrib.seq2seq.dynamic_decode(
			decoder = decoder, 
			output_time_major = False, 
			impute_finished = False, #beam search의 경우 이 부분은 False
			maximum_iterations = maximum_target_length
		)

	#decoder_output[0].shape = [batch, time, beam]
	transe_output = tf.transpose(decoder_output[0], [2,0,1]) # [beam, batch, time]
	best_output = transe_output[0] #빔 단위로 나뉘고, 그 안에서 batch 단위로 묶이니깐 0번째 쓰면 젤높은확률의 빔 + 배치별로 하나씩 다 묶임.

	return decoder_output, best_output



def greedy_decoder_for_study(output_layer, decoder_cell):
	#greedy search is equal to beam search when beam_width == 1
	helper = tf.contrib.seq2seq.GreedyEmbeddingHelper(
			embedding = embedding, 
			start_tokens = tf.tile([dic['go']], [batch_size]),
			end_token = dic['eos']
		) 
	decoder = tf.contrib.seq2seq.BasicDecoder(
			cell = decoder_cell, 
			helper=helper, 
			initial_state = en_state_concat, 
			output_layer = output_layer
		)
	decoder_output, _, _ = tf.contrib.seq2seq.dynamic_decode(  # sequence is decoder_output[1]
			decoder = decoder, 
			output_time_major = False, 
			impute_finished = True, 
			maximum_iterations = maximum_target_length
		)
	
	return decoder_output, decoder_output[1]



def testcode_greedy_equals_beam_when_beamsize_is_1(output_layer, decoder_cell):
	_, output = greedy_decoder_for_study(output_layer, decoder_cell)

	_, best_beam = beam_decoder(output_layer, decoder_cell, beam = 1) 

	test_set = read_csv(test_set_path, 'test')
	en_input, de_output = get_batch(test_set[0:10], 'test')

	print('input\n', en_input)
	print('target\n',de_output)
	
	result = sess.run(output, {x:en_input, batch_size:len(en_input)})
	print('greedy\n',result)

	result = sess.run(best_beam, {x:en_input, batch_size:len(en_input)})
	print('beam_with:1\n',result)



#### Network
target_sequence_length = tf.placeholder(tf.int32, [None], name = "target_sequence_length") 
batch_size = tf.placeholder(tf.int32, [], name = 'batch_size')

x = tf.placeholder(tf.int32, [None, None], name = 'x') #batchsize, input_length
decoder_input = tf.placeholder(tf.int32, [None, None], name = 'decoder_input') #decoder input => 'go', sequence
decoder_target = tf.placeholder(tf.int32, [None, None], name = 'decoder_target') # decoder_target => sequence, 'eos'

embedding = tf.Variable(tf.random_uniform([target_size, embedding_dimension], -1., 1.)) #if embedding lookup size is bigger than embedding . then output is 0
embedding_encoder = tf.nn.embedding_lookup(embedding, x) #target_size가 14이므로 0~13의 값만 갖는데 만약 x가 14 이상이면 00..0..00 리턴
embedding_decoder = tf.nn.embedding_lookup(embedding, decoder_input)



####encoder
fw = tf.nn.rnn_cell.LSTMCell(cell_num) # cell must write here. not in def function
bw = tf.nn.rnn_cell.LSTMCell(cell_num)
_, en_state_concat = encoder(fw, bw)



####decoder
output_layer = tf.layers.Dense(target_size) #output_layer, decoder_cell must write here. not in def function
decoder_cell = tf.nn.rnn_cell.LSTMCell(cell_num*2) # encoder is bidirectional, so decoder_cell_num is 2*encoder_cell_num

_, logits, output = train_decoder(output_layer, decoder_cell)

_, best_beam_1 = beam_decoder(output_layer, decoder_cell, 1) #beamsize= 1 , test_decoder_output[0].shape = [batch, time, beam]
_, best_beam_2 = beam_decoder(output_layer, decoder_cell, 2) #beamsize= 2 , test_decoder_output[0].shape = [batch, time, beam]
_, best_beam_3 = beam_decoder(output_layer, decoder_cell, 3) #beamsize= 3 , test_decoder_output[0].shape = [batch, time, beam]



####weight update
#https://www.tensorflow.org/versions/r1.4/api_docs/python/tf/contrib/seq2seq/sequence_loss
#masks = tf.sequence_mask([3,2,3], 5, dtype=tf.float32, name='masks') # [[1 1 1 0 0], [1.1.0.0.0], [1.1.1.0.0]]
masks = tf.sequence_mask(target_sequence_length, maximum_target_length, dtype=tf.float32, name='masks') #sequence에서 의미 있는 길이만 사용하려고 mask 활용함.
cost = tf.contrib.seq2seq.sequence_loss(logits, decoder_target, masks)
optimizer = tf.train.AdamOptimizer(learning_rate)
minimize = optimizer.minimize(cost)



####session
sess = tf.Session()
sess.run(tf.global_variables_initializer())
saver = tf.train.Saver(max_to_keep=10000)



####tensorboard
train_loss_tensorboard = tf.placeholder(tf.float32, name='train_loss')
vali_loss_tensorboard = tf.placeholder(tf.float32, name='vali_loss')
test_accuracy_tensorboard1 = tf.placeholder(tf.float32, name='test1')
test_accuracy_tensorboard2 = tf.placeholder(tf.float32, name='test2')
test_accuracy_tensorboard3 = tf.placeholder(tf.float32, name='test3')

train_summary = tf.summary.scalar("train_loss", train_loss_tensorboard)
vali_summary = tf.summary.scalar("vali_loss", vali_loss_tensorboard)
test1_summary = tf.summary.scalar("test_accuracy1", test_accuracy_tensorboard1)
test2_summary = tf.summary.scalar("test_accuracy2", test_accuracy_tensorboard2)
test3_summary = tf.summary.scalar("test_accuracy3", test_accuracy_tensorboard3)
merged = tf.summary.merge_all()
writer = tf.summary.FileWriter(tensorboard_path, sess.graph)



####run
train_set = read_csv(train_set_path)
vali_set = read_csv(vali_set_path)
test_set = read_csv(test_set_path, 'test')

run(train_set, vali_set, test_set)



#testcode
#testcode_greedy_equals_beam_when_beamsize_is_1(output_layer, decoder_cell)
