import pandas as pd
import numpy as np
from scipy.stats import randint
import math

_MIN_AGE_=18
_MAX_AGE_=78


def generate_ages(mean, sd, N, seed=42):
    """
    Generate a list of N values representing adult ages (18-78) drawn from a normal distribution 
    with given mean and sd
    """
    #note: scipy.stats.skewnorm will generate skewed distributions!
    age_values=np.random.default_rng(seed=seed).normal(loc=mean, scale=sd, size=N)
    age_values=np.clip(age_values, _MIN_AGE_, _MAX_AGE_).astype(int)
    return age_values

def generate_credit_scores(age_values, seed=42):
    """
    Generate a random credit score for a set of individuals with given ages.
    Credit scores are biased by age, such that older people tend to have higher credit scores
    """
    # Generate random values for credit score from a normal distribution
    mean_credit_score = 730
    std_dev_credit_score = 60
    n=len(age_values)
    credit_score_values = np.random.default_rng(seed=seed).normal(loc=mean_credit_score,
                                       scale=std_dev_credit_score, 
                                       size=n)
    #tip the scales so that the credit scores tend to be higher for older people
    #the youngest people have credit scores 40 points lower
    age_devs=(age_values-_MAX_AGE_)/float(_MAX_AGE_-_MIN_AGE_)*40
    credit_score_values = np.clip(credit_score_values+age_devs, 300, 900).astype(int)
    return credit_score_values

def age_count(age):
    """
    Use rules to generate a random integer that depends on age.
    This looks like the sort of thing that a decision tree would excel at
    """
    baseval=10
    #bins: <20; 20-39; 40-59; 60-79
    agebin = int(age/20)
    match agebin:
        case 0:
            #mu = x_bar - 10.5
            value=baseval-randint.rvs(10,20)
        case 1:
            #mu = x_bar + 11
            value=baseval+randint.rvs(10,20)+randint.rvs(10,20)
        case 2:
            #mu = x_bar + 21
            value=baseval+randint.rvs(12,24)+randint.rvs(12,24)+randint.rvs(12,24)
        case _:
            #mu = x_bar + 4.5
            value=baseval+randint.rvs(2,18)    
    return value

def csv_count(credit_score):
    """
    Use a function to generate a random integer that depends on credit score
    I think it should be a parabola: y=(x-mu)^2
    """
    mean_credit_score=730
    x=abs(credit_score-mean_credit_score)
    modifier=int(x**0.75)
    value=modifier+randint.rvs(1,4)
    return value

def personality_count(z_score):
    """
    Use a normally-distributed value to index an integer
    """
    #MU=5
    value=(z_score+1)*10
    if value == 0:
        valence=0
    else:
        valence = int(value/abs(value))
    if abs(value) > 30:
        return (30*valence)
    else:
        return int(value)



def generate_clicks(df):
    """
    Generate a dataframe of numbers that contribute to the number of website clicks 
    for a set of individuals with given characteristics defined in dataframe df
    """
    
    #df.apply function will apply function to each row in dataframe
    age_counts=df.apply(lambda row: age_count(row[0]), axis=1)
    csv_counts=df.apply(lambda row: csv_count(row[1]), axis=1)
    personality_counts=df.apply(lambda row: personality_count(row[2]), axis=1)
    result=pd.concat([age_counts, csv_counts, personality_counts], axis=1)
    return result


def csv_worth(csv):
    """
    Generate the part of a person's net worth that's attributable 
    to their credit score
    """
    exp=csv/680.0
    baseval=math.log(csv,2)
    value=baseval**exp
    if csv<500:
        return int(value*2000)
    else:
        return int(value*5000)
    
def age_worth(age):
    """
    Generate the part of a person's net worth attributable to age
    """
    baseval=max(age-30,5)
    bucket=math.ceil(age/20)
    exp_denominator=5.0+max(0, age-74)
    exp=(bucket/exp_denominator)
    value=baseval**exp
    return int(value*8000)

def clicks_worth(clicks):
    """
    Generate the part of the person's net worth attributable
    to exploring products
    """
    #return clicks*15+int(math.sqrt(clicks)*1000)
    return clicks*500+randint.rvs(-10,10)

def circumstance_worth():
    """
    Generate the part of the person's net worth attributable to random
    circumstances (normally distributed)
    """
    base = int(np.random.normal(loc=40000, scale=5000))
    return max(10000, base)


