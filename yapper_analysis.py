import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
plt.style.use('seaborn')

# =============================================================================
# analysis / transformation functions
# =============================================================================
def read_data():
    '''form the base dataframe by importing and joining the two data sets'''
    # bring in activity data
    act = pd.read_csv('./team_activity.csv', header = None)
    act.columns = ['ds', 'team_id', 'country', 'industry_id', 'active_users',
                   'messages_7d']
    
    # bring in industry data map
    ind = pd.read_csv('./industry_map.csv')
    
    # conform the duplicate healthcare industries to same name
    ind['industry'] = ind['industry'].str.replace('Health Care', 'Healthcare')
    
    # merge the industry data with activity data, prioritizing activity data 
    df = pd.merge(act, ind,
                  how = 'left',
                  on = 'industry_id')
    return df    
    

def check_no_active_users(df):
    '''filter out teams that had no active users for entire analysis period'''
    dau_check = df.groupby(['team_id'])['active_users'].agg(
        ['min', 'max'])
    dau_check['filter'] = dau_check['max'] + dau_check['max']
    inactive_teams = dau_check[dau_check['filter'] == 0]
    return [t for t in inactive_teams.index]

    
def check_consistent_cohort(df):
    '''only analyze teams present all days of the analysis period'''
    cohort_chk = df.groupby(['team_id'])['active_users'].count()
    cohort_chk = cohort_chk[cohort_chk != max(cohort_chk)]
    return [t for t in cohort_chk.index]


def check_missing_dates(df):
    '''check full date range against a unique set of dates present
    in the dataframe to make sure we are not missing any days - we are
    missing one: 3/13/2020'''
    dates = set(pd.to_datetime(df['ds']))
    dt_rng = pd.date_range(start = min(dates), end = max(dates), freq = 'd')
    missing_dts = [d for d in dt_rng if d not in dates]
    return missing_dts


def active_users_by_ind(df):
    '''find industries with the most active users at the end of July 2020.
    Addresses question #1'''
    # filter data to the last week of the analysis and then groupby/sum by ind
    end_jul_df = df[df['ds'] >= '2020-07-25']
    ind_users = end_jul_df.groupby(['industry'])['active_users'].sum(
            ).reset_index()
    # compute a daily average using the 7 day sums by industry
    ind_users['avg_daily_users'] =  ind_users['active_users'] / 7
    ind_users.sort_values('active_users', ascending = False, inplace = True)
    return ind_users


def avg_team_sz_top_countries(df, top_n):
    '''find the top countries by active users, and then compute the average
    team size in those countries using last month of user data as proxy.
    Addresses question #2'''
    # use last month of data to get a more stable team size measure
    cntry_df = df[df['ds'] >= '2020-07-01']
    
    # get distinct count of teams and total users by country
    cntry_summ = cntry_df.groupby(['country']).agg({'team_id':'nunique',
                    'active_users':'mean'}).reset_index()
    
    # rename columns to accurately reflect metrics computed
    cntry_summ.columns = ['country', 'countd_teams', 'avg_team_sz']
    
    # retrieve only top 5 countries and sort descending
    cntry_summ['teams_cnt_rank'] = cntry_summ['countd_teams'].rank(
            ascending = False)
    cntry_summ.sort_values('teams_cnt_rank', inplace = True)
    
    # filter to top n countries
    cntry_summ = cntry_summ[cntry_summ['teams_cnt_rank'] <= top_n]
    
    # sort by average team size for charting
    cntry_summ.sort_values('avg_team_sz', ascending = False, inplace = True)
    return cntry_summ
    

def daily_summary_figues(df):
    '''look at overall a number of different ways including total messages,
    total active users, distinct teams present, messages per user, avg team
    size. Addresses question #3'''
    
    # get distinct count of teams and total users by country
    dly_summ = df.groupby(['ds']).agg(
        {'team_id':'nunique',
        'active_users':['sum', 'mean'],
        'messages_7d':lambda x:sum(x)/1e6
        })
    
    # rename columns to reflect metrics computed
    dly_summ.columns = ['countd_teams', 'total_active_users', 'avg_team_users',
                        'total_msgs_mm']
    
    # compute messages per user
    dly_summ['msgs_per_user'] = dly_summ[
            'total_msgs_mm']*1e6/dly_summ['total_active_users']
    dly_summ = dly_summ.reset_index()    
    return dly_summ
    

# =============================================================================
# charting functions
# =============================================================================
def chart_top_industries(df):
    fig, ax = plt.subplots(figsize = (8, 5),
                           sharey = True,
                           sharex = True)
    
    df.plot(kind = 'bar', x = 'industry', y = 'avg_daily_users', width = .85,
            ax = ax)
    
    # set titles
    ax.set(title = 'Daily Active Users by Industry (7/25-7/31 Avg.)',
           xlabel = None,
           ylabel = 'Avg. Daily Active Users')
    ax.legend().set_visible(False)
    
    # y axis format
    ax.get_yaxis().set_major_formatter(matplotlib.ticker.FuncFormatter(
            lambda x, p: format(int(x), ',')))
    
    # font sizes
    ax.title.set_size(14)
    
    plt.show()


def chart_daily_team_cnt(df):
    fig, ax = plt.subplots(figsize = (8, 5),
                           sharey = True,
                           sharex = True)
    
    df.plot(kind = 'line', x = 'ds', y = 'countd_teams', ax = ax)
    
    # set titles
    ax.set(title = 'Count of Distinct Teams',
           xlabel =  None,
           ylabel = 'Distinct Teams')
    ax.legend().set_visible(False)
    
    # font sizes
    ax.title.set_size(14)
    
    # y axis format
    ax.get_yaxis().set_major_formatter(matplotlib.ticker.FuncFormatter(
            lambda x, p: format(int(x), ',')))
    
    plt.show()
    
    
def chart_daily_active_users(df):
    fig, ax = plt.subplots(figsize = (8, 5),
                           sharey = True,
                           sharex = True)
    
    df.plot(kind = 'line', x = 'ds', y = 'total_active_users', ax = ax)
    
    # set titles
    ax.set(title = 'Daily Active Users',
           xlabel = None,
           ylabel = 'Active Users')
    ax.legend().set_visible(False)
    
    # y axis format
    ax.get_yaxis().set_major_formatter(matplotlib.ticker.FuncFormatter(
            lambda x, p: format(int(x), ',')))
    
    # font sizes
    ax.title.set_size(14)
    
    plt.show()
    

def chart_avg_team_size(df):
    fig, ax = plt.subplots(figsize = (8, 5),
                           sharey = True,
                           sharex = True)
    
    df.plot(kind = 'line', x = 'ds', y = 'avg_team_users', ax = ax)
    
    # set titles
    ax.set(title = 'Avg. Team Size - Active Users',
           xlabel = None,
           ylabel = 'Avg. Active Users')
    ax.legend().set_visible(False)
    
    # y axis format
    ax.get_yaxis().set_major_formatter(matplotlib.ticker.FuncFormatter(
            lambda x, p: format(int(x), ',')))
    
    # font sizes
    ax.title.set_size(14)
    
    plt.show()
    
    
def chart_msgs_sent(df):
    fig, ax = plt.subplots(figsize = (8, 5),
                           sharey = True,
                           sharex = True)
    
    df.plot(kind = 'line', x = 'ds', y = 'total_msgs_mm', ax = ax)
    
    # set titles
    ax.set(title = 'Messages Sent Last 7 Days (mm)',
           xlabel = None,
           ylabel = 'Messages (mm)')
    ax.legend().set_visible(False)
    
    # y axis format
    ax.get_yaxis().set_major_formatter(matplotlib.ticker.FuncFormatter(
            lambda x, p: format(x, ',')))
    
    # font sizes
    ax.title.set_size(14)
    
    plt.show()
    
    
def chart_top_cntry_team_sz(df):
    fig, ax = plt.subplots(figsize = (8, 5),
                           sharey = True,
                           sharex = True)
    
    df.plot(kind = 'bar', x = 'country', y = 'avg_team_sz', width = .85,
            ax = ax)
    
    # set titles
    ax.set(title = "Avg. Team Size - Top 5 Countries by Active Users - Jul '20",
           xlabel = None,
           ylabel = 'Avg. Team Size (Month)')
    ax.legend().set_visible(False)
    
    # y axis format
    ax.get_yaxis().set_major_formatter(matplotlib.ticker.FuncFormatter(
            lambda x, p: format(x, ',')))
    
    # font sizes
    ax.title.set_size(14)
    
    plt.show()
      

# =============================================================================
# main script    
# =============================================================================
if __name__ == '__main__':
    
    # read and transform data for charting
    df = read_data()
    
    # clean up teams that had no active users for the entire period
    inactive_list = check_no_active_users(df)
    df = df[~df['team_id'].isin(inactive_list)]
    
    # clean up teams that were not present in the cohort for entire period
    cohort_chk = check_consistent_cohort(df)
    df = df[~df['team_id'].isin(cohort_chk)]
    
    # find missing dates
    missing_dts = check_missing_dates(df)
    
    # find answers to questions in README
    ind_users = active_users_by_ind(df)
    cntry_summ = avg_team_sz_top_countries(df, top_n = 5)
    dly_summ = daily_summary_figues(df)
    
    # chart data for presentation/exploration
    chart_top_industries(ind_users)
    chart_top_cntry_team_sz(cntry_summ)
    chart_daily_team_cnt(dly_summ)
    chart_daily_active_users(dly_summ)
    chart_msgs_sent(dly_summ)
    chart_avg_team_size(dly_summ)

