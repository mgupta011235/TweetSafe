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


###############################################################################
#create dataframe with only hateful comments
# build sql query to extract comments
query = []
for i in range(len(final_hateful_srs)):
    print 'Getting comments from {}'.format(final_hateful_srs[i])
    query.append("SELECT subreddit,id, name, body FROM MAY2015 WHERE subreddit = '" + final_hateful_srs[i] + "';")

# load df with the first set of results
df = pd.read_sql_query(query[0], conn)

# iterate through queries and append to dataframe;
for i in range(1, len(query)):
    print "appending query to dataframe {}".format(i)
    df = df.append(pd.read_sql_query(query[i], conn), ignore_index=True)

# Reset index, to make it workable
df.reset_index(drop=True)

#
print "Done with Hate Categorization"

# Let's save this file for later access!
pickle.dump(df, open('../../data/hateComments.p', 'wb'))

###############################################################################
#create dataframe with only not hateful comments
# build sql query to extract comments
query = []
for i in range(len(final_nothate_srs)):
    print 'Getting comments from {}'.format(final_nothate_srs[i])
    query.append("SELECT subreddit,id, name, body FROM MAY2015 WHERE subreddit = '" + final_nothate_srs[i] + "';")

# load df with the first set of results
df = pd.read_sql_query(query[0], conn)

# iterate through queries and append to dataframe;
for i in range(1, len(query)):
    print "appending query to dataframe {}".format(i)
    df = df.append(pd.read_sql_query(query[i], conn), ignore_index=True)

# Reset index, to make it workable
df.reset_index(drop=True)

#
print "Done with Not Hate Categorization"

# Let's save this file for later access!
pickle.dump(df, open('../../data/nothateComments.p', 'wb'))

# # To load file:
# df = pickle.load(open('../../data/labeledRedditComments.p', 'rb'))

# Don't forget to close the connection!!!
conn.close()
