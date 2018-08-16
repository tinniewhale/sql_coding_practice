# SQL Practice Question
###########################
### Q1 Friend Request #####
###########################
# Q1.1: calculate voerall acceptance rate of requests
# Table friend_request: sender_id | receiver_id | request_date
# Table request_accepted: requester_id | accepter_id | accept_date 

# Note: (1) NULL Accepted to 0 
#       (2) Multiple Request/Acceptance from the same person: 
#					a. multiple acceptance for the same request 
#					b. multiple request to the same person
#					c. multiple pair of requests and acceptance
# if % format needed: FORMAT((xx/xx),'P2')


SELECT c.request_date, SUM(case c.accept_date when null then 0 else 1 end) * 1.0/COUNT(*)
FROM (
	SELECT a.sender_id, a.receiver_id, MIN(a.request_date) as initial_request_date, MAX(b.accept_date) as last_accept_date
	FROM friend_request a
	LEFT JOIN request_accepted b
	ON a.sender_id = b.requester_id
	AND a.receiver_id = b.accepter_id
	AND b.accept_date >= a.request_date # accept after request date 
	# (TRUE or NULL is TRUE, FALSE or NULL is NULL)
	# If accept_date NULL, still
	GROUP BY a.sender_id, a.receiver_id
) c
GROUP BY c.request_date

# Q1.1.1: Calculate daily acceptance rate
# QUESTION: does it mean accept within the same day of request sent????
# Table request_acceptance: sender_id | receiver_id | request_date | status
# status: sent, accepted, rejected

SELECT request_date as Dates, sum(accepted) * 1.0/count(*) as Acceptance_Rate
FROM (
	SELECT sender_id, receiver_id, request_date, status, (case status when 'accepted' then 1 else 0 end) as accepted
	FROM request_acceptance
	) a
GROUP BY request_date


# Q1.2: check if the overall acceptance rate has decreased from 60% in may'17 to 30% in sep'17  
SELECT A2.Acceptance_Rate - A1.Acceptance_Rate
FROM Acceptance as A1, Acceptance as A2
# ON DATEDIFF(month, CAST(A1.Year AS VARCHAR) + '/' + CAST(A1.Month AS VARCHAR) + '/01', 
# 	CAST(A2.Year AS VARCHAR) + '/' + CAST(A2.Month AS VARCHAR) + '/01') = 4
WHERE A1.Year = 2017 AND A1.Month = 5
AND A2.Year = 2017 AND A2.Month = 9

# Q1.3 find the user with the most friends
SELECT ID_1, Count(*) as Num_of_Friends
FROM
	(SELECT DISTINCT a.sender_id as ID_1, a.receiver_id as ID_2
	FROM friend_request a
	LEFT JOIN request_accepted b
	ON a.sender_id = b.requester_id
	AND a.receiver_id = b.accepter_id
	AND b.accept_date >= a.request_date
	WHERE b.accept_date IS NOT NULL
	UNION ALL
	SELECT DISTINCT a.receiver_id as ID_1, a.sender_id as ID_2
	FROM friend_request a
	LEFT JOIN request_accepted b
	ON a.sender_id = b.requester_id
	AND a.receiver_id = b.accepter_id
	AND b.accept_date >= a.request_date
	WHERE b.accept_date IS NOT NULL) c
GROUP BY ID_1
ORDER BY Num_of_Friends DESC

#################################
### Q2 User Status Tracking #####
#################################

# Q2.1: How many users turned the feature on today
# Table feature_track (tracks every time a user turns a feature on or off): user_id | action ("on" or "off) | datetime
# Today: MYSQL - CURDATE() '2018-08-13', SQL Server - GETDATE()

SELECT COUNT DISTINCT user_id
FROM feature_track
WHERE action = "on"
AND cast(datetime, Date) = CURDATE()

# Q2.2: In a table that tracks the status of every user every day, how would you add today's data to it?
INSERT INTO tracking VALUES (user_id, date, status)

# Q2.3: Create a daily tracking table of user status
# Table tracking: user_id, status ('active', 'inactive'), date
# Table day: user_id, date
# Possible statuses: 
#			status  | yesterday | today
#			stayed  | yes       | yes
#			churned | yes       | no
#			revived | no        | yes
#			new     | Null      | yes

# Note: account for undefined status - 
#				yesterday no, today no
#				yesterday no, today null
#				yesterday yes, today null
# Question: does tracking table contain today's status???

SELECT USER, CASE
WHEN yesterday = 'inactive' AND today = 'active' THEN 'revived'
WHEN yesterday = 'active' AND today = 'active' THEN 'stayed'
WHEN yesterday = 'active' AND today = 'inactive' THEN 'churned'
WHEN yesterday = 'new' AND today = 'active' THEN 'new'
ELSE 'inactive' END as USER_STATUS
FROM (
	SELECT ISNULL(t.user_id, d.user_id) as USER, 
	CASE d.user_id WHEN IS NULL THEN 'inactive'
	ELSE 'active' END as today, ISNULL(t.status, 'new') as yesterday
	FROM tracking t
	OUTER JOIN day d
	ON t.user_id = d.user_id
	AND DATEDIFF(day, t.date, d.date) = 1
	) A

#################################
### Q3 Feature Usage Track ######
#################################

# Q3.1 Building a table with a summary of feature usage per user every day 
#.     (keep track of the last action by user and roll that up every day).
# Table feature_usage: user_id | feature | usage 


# Q3.2 Build a histogram of post reply count in SQL 
#.     (number of posts with x replies, x+1 replies, etc)
# Table post_reply: user_id | post | reply

# Question to ask: is every reply unique? 
SELECT num_reply, COUNT(*) as num_post
FROM (
	SELECT post, COUNT(*) as num_reply
	FROM post_reply
	GROUP BY post
	) A
GROUP BY num_reply
ORDER BY 1

# Q3.3 A table shows at the end of each day how many times in a user's history, 
# he/she has listened to a given song. So count is cumulative sum.
# Goal: Update this on a daily basis based on a second table that records in real time when a user listens to a given song. 
# Basically, at the end of each day, you go to this second table and pull a count of each user/song combination 
# and then add this count to the first table that has the lifetime count. 
# If it is the first time a user has listened to a given song, you won't have this pair in the lifetime table, 
# so you have to create the pair there and then add the count of the last day. 
# Table song_history: eod | user_id | song_id | count 
# Table song_listen: time | user_id | song_id
UPDATE song_history
SET 
SELECT user_id, song_id, ISNULL(num_play, 0) + ISNULL(count, 0) AS new_count
FROM song_hitory h
OUTER JOIN 
(SELECT user_id, song_id, cast(play_time as Date) as eod, COUNT(*) as num_play
FROM song_listen
GROUP BY user_id, song_id, cast(play_time as Date)) a
ON h.user_id = a.user_id
AND h.song_id = a.song_id


###############################
####### Q4 Menu sessions ######
###############################
# Q4.1 write a query to calculate average dwell time in seconds across all sessions (i.e. return one number)? 
#      Dwell time is the length of time between opening and closing the menu
# Table dwell: session | open_time | close_time
SELECT AVG(DATEDIFF(second, open_time, close_time)) as average_dwell_time
FROM dwell

# Q4.2 write a query (or queries) to get the percentage of all sessions 
#      that have both nav_menu_open and nav_menu_close?
# Table nav_events: user_id | event_time | event_name


# q4.3 Account for missing events by setting the dwell time to 60 seconds 
#.     whenever a nav_menu_close event is missing
#.     write a query to re-calculate the new average dwell time 
#.     when we default to 60 seconds of dwell time whenever nav_menu_close is missing

#######################################
####### Q5 Friend Recommendation ######
#######################################
# Write an SQL query that makes recommendations using the pages that your friends liked. 
# Assume you have two tables: a two-column table of users and their friends, 
# and a two-column table of users and the pages they liked. It should not recommend pages you already like.  

