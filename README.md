# Tennis Analysis

This project was an oppportunity to dive deep into an interesting dataset - that I didn't put together... The dataset used in this project is from [Kaggle](https://www.kaggle.com/robseidl/tennis-atp-tour-australian-open-final-2019).

## Generalised Tennis Match Analytics

Tennis_Analysis.py is an OO programme that was developed with the aim of being 'soft-coded' and easily perform the same analysis for any other tennis match dataset (with the same structure). This programme mainly performs preprocessing procedures; wading through the data and summarise it in a visual manner. It forms a custom dashboard - colocating several visualisations together. These visualisations include:

- Shot distributions
- Tornado plot for match statistics
- Points time series plot (showing each player's point accumulation over the duration of the match)

## Machine Learning Usecase

I took the opportunity to attempt to implement some machine learning algorithms to gleam information from this data set. The MLUse.py file extracts the desired features (chosen as things I thought would influence a tennis match) and then handles missing or invalid values, before converting non-numerical values to numerical. Initially, to see the effects of non-standardised data, this numerical data was then fed into a number of ML algorithms (from sklearn) with the accuracies compared and the best one taken forwards as the model to use for any further analysis. Through observing the **feature importance weightings** it was clear that the data did require standardising as features such as player location on the court were showing the highest importance values due to the fact that these columns included the largest numbers.
