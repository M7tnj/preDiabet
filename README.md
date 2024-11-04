# preDiabet
#### Video Demo:  <URL HERE>
#### Description: 
## <span style="color:blue;">Diabetes Prediction Model</span>

Diabetes is a chronic condition that affects millions of people worldwide. It occurs when the body either cannot produce enough insulin or cannot effectively use the insulin it produces, leading to elevated blood glucose levels. Managing diabetes is crucial to prevent complications such as heart disease, kidney failure, and nerve damage.

To aid in the prediction and management of diabetes, we have developed a neural network model. This model leverages machine learning techniques to predict the likelihood of diabetes based on various health metrics.

## <span style="color:green;">Overview of the Model</span>

### <span style="color:purple;">Data Collection</span>
The dataset is downloaded from a specified URL using the `requests` library and read into a pandas DataFrame.

### <span style="color:purple;">Data Preprocessing</span>
- The dataset is split into features (`X`) and the target variable (`y`), which indicates the presence of diabetes.
- Numerical features are identified and processed using a pipeline that includes imputation of missing values and standardization.
- The data is split into training and validation sets.

### <span style="color:purple;">Model Architecture</span>
- The neural network is built using TensorFlow and Keras. It consists of multiple dense layers with ReLU activation, batch normalization, and dropout for regularization.
- The final layer uses a sigmoid activation function to output a probability indicating the likelihood of diabetes.

### <span style="color:purple;">Model Training</span>
- The model is compiled with the Adam optimizer and binary cross-entropy loss function.
- Early stopping is used to prevent overfitting by monitoring the validation loss.
- The model is trained on the training data and validated on the validation data.

### <span style="color:purple;">Visualization</span>
- The training and validation loss, as well as accuracy, are plotted to visualize the model's performance over epochs.
