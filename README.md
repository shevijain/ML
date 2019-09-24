# PCA-with-KNN

Classifying the different fruit species using the PCA-based features and KNN classifier according to 5 different species from the fruits dataset. The images of the dataset are in coloured format but while reading taken in greyscale format.


Method/Approach:

1. Reading the images from the training folder for Apples, Bananas, Oranges, Pineapples and Raspberries folders.
2. Rescaling the images from 100x100 to 32x32 or else too much computation will be needed
3. Flattening the images so that it becomes 2D from 3D and spring the class labels in separate list.
4. Made the training data centered by subtracting the mean of each column with its elements.
5. Calculating the covariance of the centered matrix
6. Then calculating the Eigen Values and Eigen vectors of that matrix.
7. As higher Eigen value means higher variance means much better component to choose
the Eigen Values are sorted in reverse order and accordingly the Eigen vector is sorted.
One to one relationship between Eigen value and Eigen vector.
8. To find the optimal no. of principal components set a threshold and if the Eigen Values
pass the threshold those many no. of values are taken as principal components.
9. Feature vector is created by choosing the first 20 eigenpairs(eigenvalues, eigenvectors)
meaning 20 principal components.
10. For getting the projection matrix multiply the feature vector transpose with centered
training data thus resulting in dimension reduction i.e. 500x1024 now is projected on
500x20
11. The test data is Aldo multiplied with the feature matrix from the PCA algorithm giving
us projection matrix of test data with reduced dimension.
12. The KNN algorithm is used on the test projected matrix and projected training matrix
with taking the value of K as 7. The K value chosen using trial and error
13. So the highest no. of times a class occurs that image is classified as that class.
14. After the algorithm is finished implementing, the accuracy of the model is calculated.


The dataset can be found at https://www.kaggle.com/moltean/fruits
