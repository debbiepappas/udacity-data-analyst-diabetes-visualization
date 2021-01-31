# Pima Indian and German Diabetes Analysis 
## by Debbie Pappas


## Dataset

> ### <span style="background-color:yellow">Dataset Overview</span>

> Two datasets are used for the Diabetes Analysis obtained from :  
>   
> https://www.kaggle.com/uciml/pima-indians-diabetes-database  
> https://www.kaggle.com/johndasilva/diabetes  
> 
>  The Pima dataset consists of 768 Pima Indian women with the following health variables:   
1. pregnancies (number of pregnancies)  
2. glucoose (blood glucose level)  
3. blood pressure (diastolic blood pressure)    
4. skin thickness (measures body fat) 
5. insulin (test that measures amount of insulin for glucose absorpton)  
6. BMI (body mass index that measures body fat)  
7. diabetes pedigree function (measure of genetic influence for diabetes)   
8. age      
9. outcome (outcome of no diabetes(0) or diabetes(1) after the tests are performed )  

> The Germany dataset consists of 2000 Dutch women from a hospital in Frankfurt, Germany. The variables are exactly the same as the variables mentioned for the Pima dataset.
> 
> ### <span style="background-color:yellow">Data Wrangling</span>

>1. To explore the combined Pima and Germany datasets a column named 'ethnicity' was created.  
2. All '0' values for glucose, BMI, blood pressure, insulin, and skin thickness do not make sense and are replaced by NaN. 




## Summary of Findings

> ### <span style="background-color:yellow">Univariate Explorations</span>
> 
> The following observartions are made for Univariate Explorations of the combined Pima and German datasets :  
>   
> 1. The largest spikes in the glucose distribution are between 110 and 130 (more than 140 indicates diabetes).  
> 2. The largest spikes in the blood pressure distribution are between 70 and 80 (more than 80 indicates high blood pressure).  
> 3. The largest spikes in skin fold thickness is 30mm (normal ranage is around 23mm).  
> 4. The largest spikes for insulin are less than 50 mu U/ml (normal range is 16-166 mu U/ml).
> 5. The largest spikes for BMI are 35-38 (normal range is 18-25).   
> 6. The age range that has the largest count is between 21 and 31.  
> 7. The range with the most pregnancies is between 0 and 4.  
> 
> ### <span style="background-color:yellow">Bivariate Explorations</span>  
> 
> The correlation between the health variables are explored using regression plots :  
>   
> 1. There is a strong correlation between glucose and insulin with a correlation coefficient of 0.5663.  
> 2. The variables blood_pressure, skin_thickness, insulin, BMI, and age have the highest correlations with glucose (larger than 0.2).  
> 3. The glucose levels are higher for diabetic women than for nondiabetic women.
> 4. Glucose levels are roughly the same for Pima Indian women compared with Dutch women.  
> 
> ### <span style="background-color:yellow">Multivariate  Explorations</span>
> 
>1. Glucose, insulin and BMI are more correlated with diabetic outcome than the other variables in the dataset.
>2. The variables with the largest mean differences are glucose, skin thickness, insulin, BMI, and age between Pima and Dutch women.    
Pima has higher mean for pregnancies, insulin, and age. Pima also has a slight higher mean for the diabetes pedigree function variable.
>3. The diabetic outcome mean is slightly higher for Pima women than for Dutch women. The diabetic mean for Pima is 34.9% and that for Dutch is 34.2%. 


## Key Insights for Presentation

> ### <span style="background-color:yellow">Hypothesis Testing</span>

> The diabetic outcome mean differences between Pima and Dutch women are close to '0' which implies that they are roughly equally diabetic.
> 
> The null hypothesis test is that there is NO difference in diabetic means between Pima and Dutch women.    
The alternative hypothesis is that there is a difference.  
The calculated p-value is 0.686. Since the p-value is greater than alpha of 0.05 (confidence interval of 95%), we accept the null hypothesis.  





