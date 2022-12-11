# Code from internet (with and without modifications)

# Source: https://www.askpython.com/python/examples/display-images-using-python
# Purpose: Saving and showing sample images
from matplotlib import image as mpimg # Not Modified X 1
image = mpimg.imread(train_dir + "/Cauliflower/0009.jpg") # Modified X 15
plt.imshow(image) # Not Modified X 15

# Source: https://www.analyticsvidhya.com/blog/2021/06/confusion-matrix-for-multi-class-classification/
# Purpose: Produce heatmap style confusion matrix
import seaborn as sns # Not Modified X 1
from sklearn.metrics import confusion_matrix # Not Modified X 1
cm = confusion_matrix(df['true_string'], df['pred_string']) # Modified X 2
cm_df = pd.DataFrame(cm, index = classes, columns = classes) # Modified X 2
plt.figure(figsize = (9,9)) # Modified X 2
sns.heatmap(cm_df, annot = True) # Not Modified X 2
plt.title("Main Model Confusion Matrix") # Modified X 2
plt.ylabel("Actual Values") # Not Modified X 2
plt.xlabel("Predicted Values") # Not Modified X 2
