import feedparser
import pandas as pd  

#Hennes link list
urls = [
    'http://www.dn.se/nyheter/m/rss/',
    'https://rss.aftonbladet.se/rss2/small/pages/sections/senastenytt/',
    'https://feeds.expressen.se/nyheter/',
    'http://www.svd.se/?service=rss',
    'http://api.sr.se/api/rss/program/83?format=145',
    'http://www.svt.se/nyheter/rss.xml'
]

posts_by_url = {}  # Dictionary to store posts for each URL

for url in urls:
    x = feedparser.parse(url)  # Parse the RSS feed
    posts = x.entries  # Extract posts for this feed

    # Convert the posts to a DataFrame
    data = []
    for post in posts:
        data.append({
            'Title': post.get('title', ''),
            'Summary': post.get('summary', ''),
        })

    # Create a DataFrame for this URL's posts
    posts_by_url[url] = pd.DataFrame(data)


# Here we collect posts from all of URLs 
combined_df = pd.concat(posts_by_url.values(), ignore_index=True)

# Combine title and summary into one column
combined_df["title_and_summary"] = combined_df["Title"] +". "+ combined_df["Summary"]

# Drop the Title and Summary columns
combined_df.drop(columns=["Title", "Summary"], inplace=True)

news = combined_df['title_and_summary'].tolist()

combined_df.to_csv('all_posts.csv', index=False) # ////////////////////////////////////