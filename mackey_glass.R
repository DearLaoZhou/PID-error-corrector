library(keras)
#Load dataset
data <- read.csv("mackey_glass.csv")
head(data)

data <- data.matrix(data[1:10000, ])
#data <- data[1:2000,]
#mean <- apply(data, 2, mean)
#std <- apply(data, 2, sd)
#data <- scale(data, center = mean, scale = std)

#Divide the training set and testing set
plot(data[1:10000,],type = "l")
n <- nrow(data)
train_data <- data[1:(0.8 * n), ]
train_data <- array_reshape(train_data,dim = c(8000,1))
test_data <- data[(0.8 * n + 1):n, ]
test_data <- array_reshape(test_data,dim = c(2000,1))

sequence_length <- 3#sliding window

#train_sequences <- list()
#train_targets <- list()

train_sequences <- array(0,dim = c(7997,3,1))
train_targets <- array(0,dim=c(7997,1))


#Construct input-output data
for (i in sequence_length:(dim(train_data)[1]-1)) {
  train_data1 <- train_data[(i - sequence_length + 1):i, 1]
  train_data1 <- array_reshape(train_data1,dim=c(1,3,1))
  train_sequences[i - sequence_length + 1, , ] <- train_data1
  train_targets[(i - sequence_length + 1),] <- train_data[(i + 1), 1] #
}

#Create LSTM Model
model <- keras_model_sequential() %>%
  layer_lstm(units = 16, activation = 'relu', return_sequences = TRUE, input_shape = list(NULL, dim(data)[2])) %>%
  layer_lstm(units = 8, activation = 'relu') %>%
  layer_dense(units = 1, activation = 'linear')
summary(model)

#compile model
model %>% compile(
  optimizer = optimizer_adam(learning_rate = 0.001),
  loss = "mse"
)

train_x <- train_sequences
train_y <- train_targets

#fit model
history <- model %>% fit(
  train_x,
  train_y,
  epochs = 50,
  batch_size = 32,
  validation_split = 0.2 
)

test_sequences <- array(0,dim = c(1997,3,1))
test_targets <- array(0,dim=c(1997,1))

#Construct test data
for (i in sequence_length:(dim(test_data)[1] - 1)) {
  test_data1 <- test_data[(i - sequence_length + 1):i, 1]
  test_data1 <- array_reshape(test_data1,dim=c(1,3,1))
  test_sequences[i - sequence_length + 1, , ] <- test_data1
  test_targets[(i - sequence_length + 1),] <- test_data[(i + 1), 1]
}

#test_predictions <- model %>% predict(test_sequences)
test_predictions <- array(0,dim=c(1997,1))
test_pid1 <- rep(0,1997)
#test_pid2 <- rep(0,1997)

#PID error corrector
prev_error1 <- 0;error_sum1<-0
for (i in 1:dim(test_sequences)[1]){
  if(i<=1)test_predictions[i, 1] <- model %>% predict(array_reshape(test_sequences[i,,],dim = c(1,3,1)),verbose = 0)
  else{
    test_predictions[i, 1] <- model %>% predict(array_reshape(test_sequences[i,,],dim = c(1,3,1)),verbose = 0)
    #PID error corrector 1
    kp1=1.0;ki1=0.1;kd1=0.01;dt1 <- 0.1
    error1 <- test_targets[i-1,1] - test_predictions[i-1,1]
    P1 <- kp1*error1
    I1 <- ki1*(error1 + prev_error1)*dt1/2
    #I1 <- ki1*(error1 + error_sum1)*dt1
    D1 <- kd1*(error1-prev_error1)/dt1
    pid_output1 <- P1+I1+D1
    if(pid_output1 > 0.1) pid_output1 <- 0.1
    else if (pid_output1 < -0.1) pid_output1 <- -0.1
    test_pid1[i] <- test_predictions[i,1]+pid_output1
    prev_error1 <- error1
    #error_sum1 <- error1 + error_sum1
    #PID error corrector 2
    #kp2=1.2;ki2=0.1;kd2=0.05
    #dt2 <- 0.1;prev_error2 <- 0
    #error2 <- test_targets[i-1,2] - test_predictions[i-1,2]
    #P2 <- kp2*error2
    #I2 <- ki2*(error2 + prev_error2)*dt2/2
    #D2 <- kd2*(error2-prev_error2)/dt2
    #pid_output2 <- P2+I2+D2
    #if(pid_output2 > 0.5) pid_output2 <- 0.5
    #else if (pid_output2 < -0.5) pid_output2 <- -0.5
    #test_pid2[i] <- test_predictions[i,2]+pid_output2
    #prev_error2 <- error2
  } 
}

plot(test_pid1[2:1997],type = "l",col="red")
lines(test_predictions[2:1997,1],type = "l",col="black")
lines(test_targets[2:1997,1],col="blue")

#plot(test_pid2[2:429],type = "l",col="red")
#lines(test_predictions[2:429,2],type = "l",col="black")
#lines(test_targets[2:429,2],col="blue")

#plot(test_predictions[1:425,1],type = "l",col="red")
#lines(test_targets[1:425,1],col="blue")

#plot(test_predictions[1:425,2],type = "l",col="red")
#lines(test_targets[1:425,2],col="blue")
true_value=test_targets[2:1997,1];predict_value=test_pid1[2:1997]
rmse=sqrt(mean((true_value-predict_value)^2));
cat('RMSE:',rmse)
mae=mean(abs(true_value-predict_value));
cat('MAE:',mae)
mape=mean(abs((true_value-predict_value)/true_value));
cat('MAPE:',mape*100,'%')

e1<-abs(test_predictions[2:1997,1]-test_targets[2:1997,1])
e2<-abs(test_pid1[2:1997]-test_targets[2:1997,1])
plot(e1,type = "l",col="red")#LSTM
lines(e2,type = "l",col="black")#LSTM-PID
