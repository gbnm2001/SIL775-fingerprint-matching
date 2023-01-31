import matplotlib.pyplot as plt

method = input('Enter the algo = ')
positives = {}
negatives = {}
tfile = open(f'TP{method}.txt')
tp =0
fn = 0
tn = 0
fp = 0
for line in tfile:
    l = line.split(',')
    if(l[2] == '(True'):
        tp+=1
    else:
        fn+=1
    score = round(float(l[-1][:-2]),1)
    if(score in positives):
        positives[score]+=1
    else:
        positives[score] = 1

tfile.close()

nfile = open(f'N{method}.txt')
for line in nfile:
    l = line.split(',')
    if(l[2] == '(False'):
        tn+=1
    else:
        fp+=1
    score = round(float(l[-1][:-2]),1)
    if(score in negatives):
        negatives[score]+=1
    else:
        negatives[score] = 1

nfile.close()
positives = sorted(positives.items(), key= lambda x: x[0])
px = [x for (x,y) in positives]
py = [y for (x,y) in positives]
negatives = sorted(negatives.items(), key= lambda x: x[0])
nx = [x for (x,y) in negatives]
ny = [y for (x,y) in negatives]
if(method == 'HT'):
    title='Hough Transform'
    mr = 0.15
elif(method == 'GA'):
    title='Genetic Algorithm'
    mr = 0.25
elif(method=='CP'):
    title='Core point matching'
    mr = 0.125
plt.title(title)
plt.xlabel('Match ratio')
plt.ylabel('Number of test cases')
plt.plot(px, py, label = 'Genuine')
plt.plot(nx,ny,label='Imposter')
plt.plot([mr,mr],[0,100],linestyle = 'dotted', label='Decision boundary')
plt.legend()
plt.show()
precision =  tp/(tp+fp)
recall =  tp/(tp+fn)
accuracy = (tp+tn)/(tp+fp+fn+tn)
f1score = 2*precision*recall/(precision+recall)
print(f'TP = {tp} FN = {fn} TN = {tn} FP = {fp}')
print('PRECISION\t=\t', precision)
print('RECALL\t=\t',recall)
print('ACCURACY\t=\t', accuracy)
print('F1 score\t=\t', f1score)

