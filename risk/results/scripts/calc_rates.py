import pandas as pd

file_list = ['/mnt/beegfs/nragu/tsunami/ML4SicilyTsunami/resources/raw/eve_rate/med09159_BS_mih1.0-4.0_probs99ALL.txt',
            '/mnt/beegfs/nragu/tsunami/ML4SicilyTsunami/resources/raw/eve_rate/med09159_PS_mih1.0-4.0_probs99ALL.txt',
            '/mnt/beegfs/nragu/tsunami/ML4SicilyTsunami/resources/raw/eve_rate/med09174_BS_probs99ALL.txt',
            '/mnt/beegfs/nragu/tsunami/ML4SicilyTsunami/resources/raw/eve_rate/med09174_PS_probs99ALL.txt']

all_events = pd.DataFrame()

for f,file in enumerate(file_list):
    print(file)
    #read file and calculate mean probability for each event stored as a row( event_id, prob0....prob999)
    df = pd.read_csv(file, sep=',')
    # append first column and 0.05,0.16,0.5,0.84,0.95 quantiles to a new dataframe 
    df_rates = pd.DataFrame({'ID':df.iloc[:,0],'P05':df.iloc[:,1:].quantile(0.05,axis=1),
                        'P16':df.iloc[:,1:].quantile(0.16,axis=1),
                        'P50':df.iloc[:,1:].quantile(0.5,axis=1),
                        'P84':df.iloc[:,1:].quantile(0.84,axis=1),
                        'P95':df.iloc[:,1:].quantile(0.95,axis=1),
                        'Pmean':df.iloc[:,1:].mean(axis=1)})
                        
    #append to all_events
    all_events = pd.concat([all_events,df_rates],axis=0)
    del df, df_rates
    
#drop duplicates
all_events.drop_duplicates(subset='ID', keep='first', inplace=True)

#save to file
all_events.to_csv('/mnt/beegfs/nragu/tsunami/ML4SicilyTsunami/resources/processed/all_eventsBS_PS53550_perce_rates.txt', sep=',', index=False)
