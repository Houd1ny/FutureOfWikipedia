Originally, our problem statement was the following: 
“Find pages in Ukrainian Wikipedia that should be translated”. 
But this task is very subjective and hard to evaluate. How do we define “better” pages?

So as a result of our discussion we decided to change problem statement to:
“Prediction of page translation from its historical data”. 

So to solve our original task we find pages in Ukrainian wikipedia that most probably will be translated. 
Here we've made 2 assumptions:
 - a page should be translated because similar type of pages were translated before  
 - a page was translated from Ukrainian to English if Ukrainian page was created before the corresponding English page.

**Structure of project**

```data``` contains our final time series data with all features and data in aggregateed format for translated and untranslated pages  

```data collection``` contains all scripts related to data collection  
   - ```create_timeseries_views_revisions_contributors.ipynb``` : functions for creating timeseries for given titles (from pickle file) and few supporting functions  

```preprocessing/data_preprocessing.ipynb``` converts data in kind of time series format into an ordinary tabular format where each article is characterized by 1 row of data.  

```modeling/modeling.ipynb``` contains partitionaing data into train and test and model fitting and evaluation.   

```visualization``` contains visualizations on R for distribution of pages by their age on date of translation
 

