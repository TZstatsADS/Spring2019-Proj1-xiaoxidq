library(tm)
library(tidytext)
library(tidyverse)
library(DT)
library(vcd)
library(ngram)
library(sentimentr)
library(qdap)
library(syuzhet)
library(topicmodels)
setwd("/Users/xiaoxi/Documents/GitHub/Spring2019-Proj1-xiaoxidq/doc")
urlfile<-'https://raw.githubusercontent.com/rit-public/HappyDB/master/happydb/data/cleaned_hm.csv'
hm_data <- read_csv(urlfile)
urlfile<-'https://raw.githubusercontent.com/rit-public/HappyDB/master/happydb/data/demographic.csv'
demo_data <- read_csv(urlfile)

#convert all the letters to the lower case, and removing punctuation, numbers, empty words and extra white space
corpus <- VCorpus(VectorSource(hm_data$cleaned_hm))%>%
  tm_map(content_transformer(tolower))%>%
  tm_map(removePunctuation)%>%
  tm_map(removeNumbers)%>%
  tm_map(removeWords, character(0))%>%
  tm_map(stripWhitespace)

#stem words
stemmed <- tm_map(corpus, stemDocument) %>%
  tidy() %>%
  select(text)

#Create a tidy format of the dictionary to be used for completing stems
dict <- tidy(corpus) %>%
  select(text) %>%
  unnest_tokens(dictionary, text)

#remove stopwords
data("stop_words")
word <- c("happy","ago","yesterday","lot","today","months","month",
          "happier","happiest","last","week","past")
stop_words <- stop_words %>%
  bind_rows(mutate(tibble(word), lexicon = "updated"))
completed <- stemmed %>%
  mutate(id = row_number()) %>%
  unnest_tokens(stems, text) %>%
  bind_cols(dict) %>%
  anti_join(stop_words, by = c("dictionary" = "word"))

#match the stem with word of highest frequency
completed <- completed %>%
  group_by(stems) %>%
  count(dictionary) %>%
  mutate(word = dictionary[which.max(n)]) %>%
  ungroup() %>%
  select(stems, word) %>%
  distinct() %>%
  right_join(completed) %>%
  select(-stems)

# complete the stem and combine with the original cleaned_data
completed <- completed %>%
  group_by(id) %>%
  summarise(text = str_c(word, collapse = " ")) %>%
  ungroup()
hm_data <- hm_data %>%
  mutate(id = row_number()) %>%
  inner_join(completed)
write_csv(hm_data, "../output/processed_moments.csv")

##Sentiment Analysis

senti_data<-sel_data%>%
  select(gender,marital,parenthood,country,cleaned_hm,text)
sentence.list=NULL
for(i in 1:nrow(senti_data)){
  sentences=sent_detect(senti_data$cleaned_hm[i],
                        endmarks = c("?", ".", "!", "|",";"))
  if(length(sentences)>0){
    emotions=get_nrc_sentiment(sentences)
    word.count=word_count(sentences)
    if (length(sentences)==1){
      emotions=as.matrix(emotions)/(word.count+0.01)} else { emotions=diag(1/(word.count+0.01))%*%as.matrix(emotions)}
    sentence.list=rbind(sentence.list, 
                        cbind(senti_data[i,-5],
                              sentences=as.character(sentences), 
                              word.count,
                              emotions,
                              sent.id=1:length(sentences)
                        )
    )
  }
}

write_csv(sentence.list, "../output/sentence.csv")

##Topic Modeling

docs <- Corpus(VectorSource(sel_data$text))
dtm <- DocumentTermMatrix(docs)
dtm= removeSparseTerms(dtm, 0.99)
rowTotals <- apply(dtm , 1, sum) #Find the sum of words in each Document

dtm  <- dtm[rowTotals> 0, ]
topic_data <- sel_data[rowTotals>0,]
write.csv(topic_data,"../output/Topicswinfo.csv")
burnin <- 4000
iter <- 2000
thin <- 500
seed <-list(2003,5,63,100001,765)
nstart <- 5
best <- TRUE

#Number of topics
k <- 9

#Run LDA using Gibbs sampling
ldaOut <-LDA(dtm, k, method="Gibbs", control=list(nstart=nstart, 
                                                  seed = seed, best=best,
                                                  burnin = burnin, iter = iter, 
                                                  thin=thin))
ldaOut.topics <- as.matrix(topics(ldaOut))
write.csv(ldaOut.topics,"../output/DocsToTopics.csv")

#top 20 terms in each topic
ldaOut.terms <- as.matrix(terms(ldaOut,20))
write.csv(ldaOut.terms,"../output/TopicsToTerms.csv")

#probabilities associated with each topic assignment
topicProbabilities <- as.data.frame(ldaOut@gamma)
write.csv(topicProbabilities,"../output/TopicProbabilities.csv")