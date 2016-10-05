''' Go from the May2015 reddit comments sqlite database and extract & save a subset for later use '''

import sqlite3
import pandas as pd
import cPickle as pickle


# Set up connection to database
sqlite_file = '../../data/RedditMay2015Comments.sqlite'
conn = sqlite3.connect(sqlite_file)
c = conn.cursor()

# Create list of relevant subreddits, both hateful and not hateful.
# List of not hateful subreddits
final_nothate_srs = ['politics', 'worldnews', 'history', 'blackladies', 'lgbt',
                     'TransSpace', 'women', 'TwoXChromosomes', 'DebateReligion',
                     'religion', 'islam', 'Judaism', 'BodyAcceptance', 'fatlogic'
                     'gaybros','AskMen','AskWomen']
# List of hateful subreddits
final_hateful_srs = ['CoonTown', 'WhiteRights', 'Trans_fags', 'SlutJustice',
                     'TheRedPill', 'KotakuInAction', 'IslamUnveiled', 'GasTheKikes',
                     'AntiPOZi', 'fatpeoplehate', 'TalesofFatHate','hamplanethatred',
                     'shitniggerssay','neofag','altright']

all_srs = final_hateful_srs + final_nothate_srs

# build sql query to extract comments
query = []
for i in range(len(all_srs)):
    print 'Getting comments from {}'.format(all_srs[i])
    query.append("SELECT subreddit,id, name, body FROM MAY2015 WHERE subreddit = '" + all_srs[i] + "';")

# load df with the first set of results
df = pd.read_sql_query(query[0], conn)

# iterate through queries and append to dataframe;
for i in range(1, len(query)):
    print "appending query to dataframe {}".format(i)
    df = df.append(pd.read_sql_query(query[i], conn), ignore_index=True)

# Reset index, to make it workable
df.reset_index(drop=True)


# Create a not hate label for all entries & write over the hateful labels later
df['label'] = 0

for subreddit in final_hateful_srs:
    df.ix[(df.subreddit == subreddit), 'label'] = 1

# # Need to label our comments depending on subreddit.
# df.ix[(df.subreddit == 'CoonTown'), 'label'] = 'RaceHate'
# df.ix[(df.subreddit == 'WhiteRights'), 'label'] = 'RaceHate'
# print "Done with RaceHate"
# df.ix[(df.subreddit == 'Trans_fags'), 'label'] = 'GenderHate'
# df.ix[(df.subreddit == 'SlutJustice'), 'label'] = 'GenderHate'
# df.ix[(df.subreddit == 'TheRedPill'), 'label'] = 'GenderHate'
# df.ix[(df.subreddit == 'KotakuInAction'), 'label'] = 'GenderHate'
# print "Done with GenderHate"
# df.ix[(df.subreddit == 'IslamUnveiled'), 'label'] = 'ReligionHate'
# df.ix[(df.subreddit == 'GasTheKikes'), 'label'] = 'ReligionHate'
# df.ix[(df.subreddit == 'AntiPOZi'), 'label'] = 'ReligionHate'
# print "Done with ReligionHate"
# df.ix[(df.subreddit == 'fatpeoplehate'), 'label'] = 'SizeHate'
# df.ix[(df.subreddit == 'TalesofFatHate'), 'label'] = 'SizeHate'
#
print "Done with Hate Categorization"


# Let's save this file for later access!
pickle.dump(df, open('../../data/labeledRedditComments2.p', 'wb'))

# # To load file:
# df = pickle.load(open('../../data/labeledRedditComments.p', 'rb'))

# Don't forget to close the connection!!!
conn.close()
