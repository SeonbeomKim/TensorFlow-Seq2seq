import random
import csv

plus_dup_check = {}
minus_dup_check = {}

#plus 
while len(plus_dup_check) < 200000:
	first_num = random.randint(1, 99999)
	second_num = random.randint(1, 99999)	
	key = str(first_num)+'+'+str(second_num) + '=' + str(first_num + second_num)
	plus_dup_check[key] = 1

#minus
while len(minus_dup_check) < 200000:
	first_num = random.randint(1, 99999)
	second_num = random.randint(1, 99999)	
	if first_num < second_num: # 큰 수에서 작은수 빼도록함.
		first_num, second_num = second_num, first_num
	key = str(first_num)+'-'+str(second_num) + '=' + str(first_num - second_num)
	minus_dup_check[key] = 1

#print(minus_dup_check)
plus_keys = list(plus_dup_check.keys())
minus_keys = list(minus_dup_check.keys())

train = [] #200000
vali = [] # 50000
test = [] # 150000

train.extend(plus_keys[:100000])
train.extend(minus_keys[:100000])
vali.extend(plus_keys[100000:125000])
vali.extend(minus_keys[100000:125000])
test.extend(plus_keys[125000:])
test.extend(minus_keys[125000:])

random.shuffle(train)
random.shuffle(vali)
random.shuffle(test)
print(len(train), len(vali), len(test))


with open("train_set.csv", 'w', newline='') as o:
	wr = csv.writer(o)
	for i in train:
		wr.writerow(i)

with open("vali_set.csv", 'w', newline='') as o:
	wr = csv.writer(o)
	for i in vali:
		wr.writerow(i)

with open("test_set.csv", 'w', newline='') as o:
	wr = csv.writer(o)
	for i in test:
		wr.writerow(i)








#print(first_num, second_num)



#if first_len < second_len:
#	fisrt_len, second_len = second_len, first_len


