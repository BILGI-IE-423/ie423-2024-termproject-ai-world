
## *Director Prediction Based on the Film's Characteristics*


## *Principal Goals of the Project* 
The film industry is a complex environment where various factors influence the decisions behind movie production. One crucial aspect is the director, whose vision and style can impact the outcome of a film.The main purpose of the project is to predict the director by taking into account various features of the films. When generating predictions, factors including the movie's genre, cast, release year, length, type, rating, and description of the film are considered. In addition to these characteristics, Bechdel test scores and movie posters are also utilized to predict the director. These features have been considered to add new perspectives to predicting the film director.By incorporating a diverse range of features, the aim is to build a comprehensive model that can accurately predict the director of a given film.

## *Resarch Questions* 
- How accurately can we predict the director of a movie based on features such as cast, duration, genre, Bechdel score, IMDb score?
- What are the most significant factors influencing the choice of director for a movie?
- How does the performance of different machine learning algorithms compare in predicting movie directors based on film characteristics?


## *Utilizing of the datasets*
First of all, two data sets contining Amazon Prime and Netflix movies and TV shows are used. These data sets provide details on the movie's titles, directors, casts, countries, and genres.

In addition, the data set containing Bechdel tests scores of movies is used. 

A separate data set is used to obtain the necessary information about the posters of the movies.

## *Some of the Utilized Libraries*
The libraries pandas, numpy, seaborn, matplotlib, sklearn, sklearn, TensorFlow Keras, PIL, io,  have been used in this project. 

## *Preprocessing Steps*

### Detalied Explanation of Preprocessing Parts

Initially, URLs containing CSV files were fetched, read, and converted into dataframes. Subsequently, dataframe names were changed to more meaningful ones.
A for loop was then created to split the "Title" column and extract the year enclosed in parentheses. Additionally, "Title" entries, which start with a capital letter, were corrected to start with a lowercase "title".The dataframes netflix_df, prime_df, bechdel_df, and movieposter_df were merged. Null and duplicate data were regularly checked for and removed if any existed. The Netflix and Amazon datasets were merged based on the 'title' column, combining the datasets into one. Following this, three DataFrames were merged: merged_df and bechdel_df were merged based on the 'title' column into merged_bnp_df, and then merged_bnp_df and movieposter_df were merged based on the 'title' column into allmerged_df. Subsequently, the presence of duplicates in the merged data frame was checked. unwanted columns were removed from the dataset, such as 'show_id', 'date_added', 'imdbid', ', 'description',  'Imdb Link''year_y'.Then, the remaining columns were sorted in the desired order. Null data was initially examined, with 17 null values identified in the 'cast' column, 317 in the 'country' column, and 92 in the 'director' column. Following this assessment, subsequent operations were performed on the dataset. 

In the next step, the first three actors' names from each row in the 'cast' column were extracted and labeled as cast1, cast2, and cast3, while the original 'cast' column was removed. Any null values identified in the preceding steps were then removed.The code in one part splits each entry in the 'Genre' column using the '|' character and selects the first category. Similarly, it separates each entry in the 'country' column using the ',' character and picks the first country mentioned.Another part ensures that only the initial numerical value is retained within each entry of the 'duration' column. There's a section intended to identify duplicate data within the dataset, revealing 18 instances of duplicate data upon examination.And then, duplicate rows are removed from the 'cleaned_df' DataFrame. 

The following step in preprocessing involves checking for outliers.The 'Duration' column was examined for outliers, and the data points with a value equal to 1, namely [136, 169, 221, 229, 246, 247], were removed from the dataset. 

<div align="center">
    <img src="https://github.com/BILGI-IE-423/ie423-2024-termproject-ai-world/assets/162442906/f0b7f6fc-5bdf-400e-8308-0ac4d33784e5" alt="Ekran görüntüsü 2024-06-18 145254" width="700">
</div>
<div align="left">

The function f.get_ydata was used. No outlier data was found for the Bechdel test, except for the 'Duration' column. As data dropping operations are performed, inconsistencies arise with the indices. Therefore, to prevent this, the index count is reset. 



### **RGB** 

Following this, for each movie in the 'no_outlier_df' DataFrame, it calculates the normalized average of RGB components. This represents the percentage of each color in the image. These percentages are then added to new columns named 'Red', 'Green', and 'Blue'. If accessing an image fails, an error counter is incremented. Finally, the updated DataFrame, containing the RGB components of each image, is printed. 

### **VGG16**

VGG16 ( Visual Geometry Group) is a object detection and classification algorithm which is used for image object detection, image classification and facial recognition. We utilized VGG16 to extract features from posters and cluster them based on similarities. While clustering, we determined the number of clusters using elbow method as 6. 



From inspection, it can be seen that in cluster 1 posters with seas and blue colour are together, in cluster 2 main focus of the poster is human figures.  In cluster 3 posters are usually black and White with bigger text sizes than cluster 1 and 2. In cluster 4 posters are usually yellow and orange with smaller text sizes.  In cluster 5 main focus of the poster is once again human figures but now text size, colors and figure placement in the posters are different. And finally in the last cluster, posters get more colorful with green, pink, and red colours.

Some posters selected according to the clusters created are listed below.

- Cluster 1
  
  ![WhatsApp Image 2024-06-18 at 18 43 38](https://github.com/BILGI-IE-423/ie423-2024-termproject-ai-world/assets/162442906/b4966e5b-a512-4067-af65-61786ba267c5)
  ![WhatsApp Image 2024-06-18 at 18 43 39](https://github.com/BILGI-IE-423/ie423-2024-termproject-ai-world/assets/162442906/f56822b1-4fea-4293-af14-5135cc0b6c6e)
  ![WhatsApp Image 2024-06-18 at 18 43 39 (1)](https://github.com/BILGI-IE-423/ie423-2024-termproject-ai-world/assets/162442906/0b8191a8-5db5-4d72-88d0-4005e33130b4)
  
- Cluster 2 

  ![WhatsApp Image 2024-06-18 at 18 44 15](https://github.com/BILGI-IE-423/ie423-2024-termproject-ai-world/assets/162442906/42770473-a38b-4d53-8bf2-aed26a3719e0)
  ![WhatsApp Image 2024-06-18 at 18 44 15 (2)](https://github.com/BILGI-IE-423/ie423-2024-termproject-ai-world/assets/162442906/a15d4c4a-0fd2-45f4-908b-fd5bf6753422)
  ![WhatsApp Image 2024-06-18 at 18 44 15 (3)](https://github.com/BILGI-IE-423/ie423-2024-termproject-ai-world/assets/162442906/9a6fc63b-032b-4944-b9f2-544121e3703c)

- Cluster 3
    
  ![WhatsApp Image 2024-06-18 at 18 44 45](https://github.com/BILGI-IE-423/ie423-2024-termproject-ai-world/assets/162442906/3bd5909e-01ea-4d29-9461-c4dc99669818)
  ![WhatsApp Image 2024-06-18 at 18 44 46](https://github.com/BILGI-IE-423/ie423-2024-termproject-ai-world/assets/162442906/a983c810-f1af-43c5-864e-6be8edfc4595)
  ![WhatsApp Image 2024-06-18 at 18 44 46 (1)](https://github.com/BILGI-IE-423/ie423-2024-termproject-ai-world/assets/162442906/106c61eb-a05c-4c7d-ab9a-785963546b62)

 - Cluster 4
    
    ![WhatsApp Image 2024-06-18 at 18 45 16](https://github.com/BILGI-IE-423/ie423-2024-termproject-ai-world/assets/162442906/7fff6b1c-ea2b-4555-94a8-effce0b913f3)
    ![WhatsApp Image 2024-06-18 at 18 45 16 (1)](https://github.com/BILGI-IE-423/ie423-2024-termproject-ai-world/assets/162442906/36df956c-d68d-4584-b085-3956ad1b748b)
    ![WhatsApp Image 2024-06-18 at 18 45 16 (2)](https://github.com/BILGI-IE-423/ie423-2024-termproject-ai-world/assets/162442906/53e465a4-6aea-4ea0-b49a-091ea0c7d76b)

 - Cluster 5

     ![WhatsApp Image 2024-06-18 at 18 45 54](https://github.com/BILGI-IE-423/ie423-2024-termproject-ai-world/assets/162442906/66dedb71-3caa-4487-9bf6-8b4cdebc4aec)
    ![WhatsApp Image 2024-06-18 at 18 45 54 (1)](https://github.com/BILGI-IE-423/ie423-2024-termproject-ai-world/assets/162442906/d3b40cb0-01e0-4f2c-9517-0bd2e7f78e7d)
    ![WhatsApp Image 2024-06-18 at 18 45 54 (2)](https://github.com/BILGI-IE-423/ie423-2024-termproject-ai-world/assets/162442906/86bdb999-14eb-4893-868b-755c2dedac8f)

  - Cluster 6
    
    ![WhatsApp Image 2024-06-18 at 18 46 24](https://github.com/BILGI-IE-423/ie423-2024-termproject-ai-world/assets/162442906/fcc3fd90-7e3e-468d-86a7-3a2e90ffa775)
    ![WhatsApp Image 2024-06-18 at 18 46 24 (1)](https://github.com/BILGI-IE-423/ie423-2024-termproject-ai-world/assets/162442906/0146bcec-7c32-4503-b775-84a622131719)
    ![WhatsApp Image 2024-06-18 at 18 46 24 (2)](https://github.com/BILGI-IE-423/ie423-2024-termproject-ai-world/assets/162442906/1f169dfb-c910-48f6-a91f-1ba50a66c6ee)
    
### **One-hot-encoding**

The next stage after removing the poster column is the encoding stage. Encoding has been performed using OneHotEncoder. By selecting the 'Cast1', 'Cast2', and 'Cast3' columns, encoding has been done based on the 'cast' column. Before the encoding process began, the shape of the DataFrame was [576 rows × 17 columns]. After the initial encoding step, it reached [644 rows × 1457 columns].Following the encoding process based on the 'cast' column, encoding was then performed for the 'type', 'country', 'rating_merged', and 'Genre' columns. Subsequently, null values were dropped, resulting in the dataset becoming 508 rows × 1526 columns.

### **Label Encoding**

In addition to the previous encoding steps, a LabelEncoder process was also applied to the 'director' column. We also applied label encoding for clusters created based on similarity with features coming out of vgg16.

***
Since there was not enough data for each director at this stage, this posed a problem in finding the appropriate model. For this reason, directors were divided into clusters according to the number of occurrences in the data set. In the data set, directors with 1 or 2 films were divided into class0, directors with 3 or 4 films were divided into class1, and similarly, directors with 5 or 6 films were divided into class2. After completing this process, label coding was applied to the newly formed clusters.

<div align="center">
    <img src="https://github.com/BILGI-IE-423/ie423-2024-termproject-ai-world/assets/162442906/87c2b488-d261-4cfd-b350-cca8e0a231c1" alt="WhatsApp Image 2024-06-18 at 18 39 37" width="800">
</div>
<div align="left">


T-SNE, a nonlinear data reduction technique that takes multidimensional data and uses it to represent the original data in two dimensions while preserving the original high-dimensional space between datasets, was used. In this way, a data set that separates directors according to the number of films was observed. You can see this in the following graph. 

<div align="center">
    <img src="https://github.com/BILGI-IE-423/ie423-2024-termproject-ai-world/assets/162442906/3c27d03c-e11b-43ce-b3fe-494066859654" alt="Ekran görüntüsü 2024-06-18 173920" width="800">
</div>
<div align="left">

Additionally, the following graphs were generated to visualize the distribution of various features in the dataset.

 <div align="center">
  <img src="https://github.com/BILGI-IE-423/ie423-2024-termproject-ai-world/assets/162442906/5a308b33-20a2-4053-99e5-106afb925c00" alt="Ekran görüntüsü 2024-06-18 201201" width="1200">
 </div>

----

 <div align="center">
  <img src="https://github.com/BILGI-IE-423/ie423-2024-termproject-ai-world/assets/162442906/c1c86d62-49ad-43d8-b5de-7c46c904bc52" alt="Ekran görüntüsü 2024-06-18 201319" width="1200">
 </div>

-----


 <div align="center">
  <img src="https://github.com/BILGI-IE-423/ie423-2024-termproject-ai-world/assets/162442906/5cf165ff-6e01-4431-b5a2-f6826489ef80" alt="Ekran görüntüsü 2024-06-18 201240" width="800">
 </div>

-----

   <div align="center">
    <img src="https://github.com/BILGI-IE-423/ie423-2024-termproject-ai-world/assets/162442906/9c73f6e6-0766-41d9-a395-ec4bd6fe2d92" alt="Ekran görüntüsü 2024-06-18 201253" width="700">
 </div>

-----
 <div align="center">
<img src="https://github.com/BILGI-IE-423/ie423-2024-termproject-ai-world/assets/162442906/1376500d-b166-46eb-a144-8a44a75b0c56" alt="Ekran görüntüsü 2024-06-18 090058" width="800">
 </div>

-----

<div align="center">
<img src="https://github.com/BILGI-IE-423/ie423-2024-termproject-ai-world/assets/162442906/39742f30-d0eb-42e0-bc3e-5ef437cbca0f" alt="Ekran görüntüsü 2024-06-18 090111" width="800"
 </div>

<div align="left">



#### You can click the link below for interactive graphs
[Interactive Graphs](https://app.powerbi.com/groups/me/reports/e04c71c1-601f-4fc9-b530-55bd7706bbba?ctid=da2fd848-1e41-4299-b410-7b1ec11c469b&pbi_source=linkShare)

## **Model Training**

During the model training phase, director class columns were mapped to Y and the remaining columns were mapped to X, and the data was divided into 80 percent and 20 percent train-test sets. Then, min max scaler was used to scale the data.
In the following stage, logistic regression, SVM, kNN, AdaBoost, and Neural Network algorithms were tried for various hyperparameters and the best result was sought.
The best parameters and average scores found are as follows.
- Best Mean Parameters for Logistic Regression : {'C': 0.1, 'Class Weight': None} with a mean score of 0.8744
- Best Mean Parameters for SVM :  {'C': 0.5, 'Kernel': 'linear', 'Class Weight': 'balanced'} with a mean score of 0.8744
- Best Mean Parameters for kNN :  {'neighbors': 9, 'Class Weight': 'uniform'} with a mean score of 0.8744
- Best Mean Parameters for AdaBoost : {'n_estimators': 50, 'learning_rate': 0.01} with a mean score of 0.8744
- Best Mean Parameters for Neural Network : {'Hidden Layers': (50,), 'Activation': 'tanh', 'Alpha': 0.01} with a mean score of 0.8719

According to the above results, the most suitable model was determined as SVM and therefore the SVM model was used in the determination phase.

## **Evaluation of Model Performance**

We used Accuracy, Precision, Recall metrics and ROC curve methods to evaluate model performance. 
- Accuracy: Accuracy is the ratio of examples that the model predicted correctly to the total predictions and it  is equal to 0.9020 for our test set.
- Precision: Precision is the rate at which the samples predicted by the model to be positive are actually positive, and the value of this metric for the test set is equal to 0.3036 as a macro average.
- Recall: Recall measures how much of the true positive samples the model can accurately predict and was measured at 0.3297 for the test set.
- ROC & AUC: ROC is a probability curve and the area under it, AUC, represents the degree or measure of separability. As the area under the curve increases, the discrimination performance between classes increases. These areas were found to be 0.52 for class 0, 0.59 for class 1, and 0.94 for class 2.

  The created ROC curves are shown below.

<div align="center">
    <img src="https://github.com/BILGI-IE-423/ie423-2024-termproject-ai-world/assets/162442906/74bb2c93-c521-418b-9916-cd25196ac35e" alt="Ekran görüntüsü 2024-06-18 210041" width="800">
</div>
<div align="left">

## **Future Works**

Since the data is limited and the applied model cannot directly predict the director, more data can be collected, a more convenient data set can be created and a new model that directly predicts the director can be developed.

----

## **Project Timeline**

```mermaid

gantt
       dateFormat  YYYY-MM-DD
       title IE 423 Project Time Line

       section Research
       Data Sets Rewiev                                   :done, des1, 2024-03-03,7d
       Additional Data Set Finding                        :done, des2, 2024-03-11,6d
       Determining the Research Questions                 :done, des3, 2024-03-18,15d


       section Preprocessing
       Merging Data Sets                                  :done, des6, 2024-04-5, 1d
       Handling Missing,Outlier, Duplicate Data           :done, des7, 2024-04-6, 4d
       Encoding of Categorical Data                       :done, des8, 2024-04-11, 4d
       Image Preprocessing(RGB)                           :done, des9, 2024-04-16, 5d
       Image Preprocessing(VGG16)                         :done, des9, 2024-04-22, 5d
       Visualization                                      :done, des10, 2024-04-28,2d



       section Modeling
       Splitting train/test datasets                      :done, des11, 2024-05-06, 2d
       Scaling                                            :done, des12, 2024-05-08, 2d
       Model Training                                     :done, des13, 2024-05-10, 5d
       Model Determination                                :done, des14, 2024-05-15, 3d
       Performance Measurement                            :done, des15, 2024-05-18, 5d
       Model Testing                                      :done, des16, 2024-05-22, 5d

       section Finalizing
       Final Checks                                       :done, des17, 2024-06-01,5d
       Publishing                                         :done, des18, 2024-06-18, 1d
    

```


### *Sources of Data Sets*
[Amazon Prime Movies and TV Shows](https://www.kaggle.com/datasets/shivamb/amazon-prime-movies-and-tv-shows)

[Netflix Movies and TV Shows](https://www.kaggle.com/datasets/shivamb/netflix-shows)

[Movie Bechdel Test Scores](https://www.kaggle.com/datasets/alisonyao/movie-bechdel-test-scores?select=Bechdel_detailed.csv)

[Movie Genre from its Poster](https://www.kaggle.com/datasets/neha1703/movie-genre-from-its-poster)


#### *Contributors*

Burcu Ağu 116203010

Canan Selek 120203050

Özge Sıla Çakmak 120203055

