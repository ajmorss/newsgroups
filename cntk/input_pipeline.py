data = np.sin(np.linspace(0,96*np.pi,40000))
lookback = 2000
lookahead = 100
train_size = 1000
test_size = 300
batch_size = 20
epochs = 100
X_train = [data[i*10:i*10+lookback] for i in range(train_size)]
Y_train = [data[i*10+lookback:i*10+lookback+lookahead] for i in range(train_size)]
X_test = [data[(i+train_size)*10:(i+train_size)*10+lookback] for i in range(test_size)]
Y_test = [data[(i+train_size)*10+lookback:(i+train_size)*10+lookback+lookahead] for i in range(test_size)]

X_train = np.array(X_train)
X_train = np.expand_dims(X_train, 2)
X_test = np.array(X_test)
X_test = np.expand_dims(X_test, 2)
X_train = X_train[:,::10,:]
X_test = X_test[:,::10,:]

Y_train = np.array(Y_train)
Y_test = np.array(Y_test)
