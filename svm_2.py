from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler 
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler 
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
import pickle

with open("save_count_id_df/3_gram_waka_count_id.pkl", mode="rb") as f:
    df = pickle.load(f)
with open("save_count_id_df/3_gram_tanka_count_id.pkl", mode="rb") as f:
    idf = pickle.load(f)

x = []
y = []

for d in df.values:
    x.append(d)
    y.append(0)
for i in idf.values:
    x.append(i)
    y.append(0)
print('len', len(y))
print('len', len(x))

input_train, input_test, output_train, output_test = train_test_split( x, y, test_size=0.3)

sc = StandardScaler()
sc.fit(input_train)
input_train_std = sc.transform(input_train)
input_test_std = sc.transform(input_test)

svc_model = SVC(kernel='linear', random_state=None)

# 学習
svc_model.fit(input_train_std, output_train)

pred_train = svc_model.predict(input_train_std)
accuracy_train = accuracy_score(output_train, pred_train)
print('traning data accuracy： %.2f' % accuracy_train)

pred_test = svc_model.predict(input_test_std)
accuracy_test = accuracy_score(output_test, pred_test)
print('test data accuracy： %.2f' % accuracy_test)