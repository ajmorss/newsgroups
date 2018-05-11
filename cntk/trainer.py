model_lr_factory = create_model_mn_factory()

x = C.sequence.input_variable(1)

z = model_lr_factory(x)

l = C.input_variable(lookahead, dynamic_axes=z.dynamic_axes, name="y")

# the learning rate
learning_rate = 0.015
lr_schedule = C.learning_rate_schedule(learning_rate, C.UnitType.minibatch)
def absolute_error(z, l):
  return C.reduce_mean(C.abs(z - l))
# loss function
loss = absolute_error(z, l)

# use squared error to determine error for now
error = absolute_error(z, l)

# use adam optimizer
momentum_time_constant = C.momentum_as_time_constant_schedule(20 / -np.log(0.9)) 
learner = C.fsadagrad(z.parameters, 
                      lr = lr_schedule, 
                      momentum = momentum_time_constant)
trainer = C.Trainer(z, (loss, error), [learner])
loss_summary = []

#start = time.time()
for epoch in range(0, epochs):
    for i in range(train_size//batch_size):
        trainer.train_minibatch({x: X_train[batch_size*i:batch_size*(i+1),:,:].astype(np.float32),
                                 l: Y_train[batch_size*i:batch_size*(i+1),:].astype(np.float32)})
        
    if epoch % (epochs // 10) == 0:
        training_loss = trainer.previous_minibatch_loss_average
        loss_summary.append(training_loss)
        print("epoch: {}, loss: {:.4f}".format(epoch, training_loss))

errs = []
for i in range(test_size//batch_size):
    errs = trainer.test_minibatch({x: X_test[batch_size*i:batch_size*(i+1),:,:].astype(np.float32),
                                   l: Y_test[batch_size*i:batch_size*(i+1),:].astype(np.float32)})
print(np.mean(errs))
        
from matplotlib import pyplot as plt

plt.plot(z.eval({x: X_train[11,:,:]})[0], '.')
plt.plot(Y_train[11,:], 'x')
plt.show()
