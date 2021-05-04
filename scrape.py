import json
import sys
from urllib.request import urlopen
import time
import numpy

class Scraper:

    def __init__(self, subreddit):
        self.lowestTime = 99999999999999999
        self.subReddit = subreddit
        self.allPosts = []

    def redditSampler(self, numPosts: int, start = int(time.time()) - 7*86400) -> list:
        toReturn = []
        while (numPosts > 0):
            toReturn.extend(self.sample100(start))
            start = self.lowestTime-1
            numPosts -= 100
        return toReturn

    def sample100(self, start: int):
        url = "https://api.pushshift.io/reddit/search/submission/?subreddit=" + self.subReddit
        url += "&size=100&score=%3E10&before=" + str(start)
        print(url)
        f = urlopen(url)
        innerContents = f.read()
        parsed = json.loads(innerContents)
        self.lowestTime = min(self.lowestTime, numpy.min([int(post["created_utc"]) for post in parsed["data"]]))
        return parsed["data"]

    def save(self):
        allPosts = [post for post in self.allPosts if "selftext" in post and "locked" in post and
                    post["selftext"] != "[deleted]" and post["locked"] == False]
        for post in allPosts:
            post.pop('all_awardings', None)
            post.pop('allow_live_comments', None)
            post.pop('author', None)
            post.pop('author_flair_css_class', None)
            post.pop('author_flair_richtext', None)
            post.pop('author_fullname', None)
            post.pop('author_patreon_flair', None)
            post.pop("author_flair_template_id", None)
            post.pop("author_flair_text_color", None)
            post.pop('author_premium', None)
            post.pop('awarders', None)
            post.pop('can_mod_post', None)
            post.pop('contest_mode', None)
            post.pop('domain', None)
            post.pop('full_link', None)
            post.pop('gallery_data', None)
            post.pop('gildings', None)
            post.pop('id', None)
            post.pop('is_crosspostable', None)
            post.pop('is_gallery', None)
            post.pop('is_meta', None)
            post.pop('is_original_content', None)
            post.pop('is_reddit_media_domain', None)
            post.pop('is_robot_indexable', None)
            post.pop('is_self', None)
            post.pop('is_video', None)
            post.pop('link_flair_background_color', None)
            post.pop('link_flair_text_color', None)
            post.pop('link_flair_type', None)
            post.pop('link_flair_richtext', None)
            post.pop('media_metadata', None)
            post.pop('media_only', None)
            post.pop('no_follow', None)
            post.pop('num_comments', None)
            post.pop('num_crossposts', None)
            post.pop('over_18', None)
            post.pop('parent_whitelist_status', None)
            post.pop('permalink', None)
            post.pop('pinned', None)
            post.pop('pwls', None)
            post.pop('retrieved_on', None)
            post.pop('send_replies', None)
            post.pop('spoiler', None)
            post.pop('stickied', None)
            post.pop('subreddit', None)
            post.pop('subreddit_id', None)
            post.pop('subreddit_subscribers', None)
            post.pop('subreddit_type', None)
            post.pop('suggested_sort', None)
            post.pop('thumbnail', None)
            post.pop('thumbnail_height', None)
            post.pop('thumbnail_width', None)
            post.pop('total_awards_received', None)
            post.pop('treatment_tags', None)
            post.pop('upvote_ratio', None)
            post.pop('url', None)
            post.pop('url_overridden_by_dest', None)
            post.pop('whitelist_status', None)
            post.pop('wls', None)
            post.pop('post_hint', None)
            post.pop('preview', None)
            post.pop('author_flair_type', None)
            post.pop('crosspost_parent', None)
            post.pop('crosspost_parent_list', None)
            post.pop("link_flair_template_id", None)
        jsonStr = json.dumps(allPosts)
        text_file = open(self.subReddit + ".txt", "w")
        n = text_file.write(jsonStr)
        text_file.close()

    def load(self):
        text_file = open(self.subReddit + ".txt", "r")
        self.allPosts = json.load(text_file)
        text_file.close()
        self.lowestTime = min(self.lowestTime, numpy.min([int(post["created_utc"]) for post in self.allPosts]))

    def add1000(self):
        self.load()
        if (len(self.allPosts) == 0):
            self.allPosts.extend(self.redditSampler(1000))
        else:
            self.allPosts.extend(self.redditSampler(1000, self.lowestTime - 1))

        self.save()
        self.load()

    def getPosts(self):
        return self.allPosts

if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Incorrect Syntax. Please run scrape.py <subreddit name> <max number of posts>")

    main = Scraper(sys.argv[1])
    main.load()
    print("Current Length: {}".format(len(main.getPosts())))
    while len(main.getPosts()) < int(sys.argv[2]):
        main.allPosts = [post for post in main.getPosts() if
                    "locked" in post and "selftext" in post and "created_utc" in post and post[
                        "selftext"] != "[deleted]" and post["locked"] == False]

        main.add1000()
        print("Current Length: {}".format(len(main.getPosts())))