---
layout: post
title:  "UFC Bot PoC. Step 2. Technical side. Pinecone + Langchain + GPT3."
date:   2023-02-05 16:37:50 +0100
---

In previous [post](https://marcinnis.github.io/2023/02/04/research.html){:target="_blank"}{:rel="noopener noreferrer"} I described first tests of our UFC Bot Proof-of-Concept.

As reminder goal was to see if it's possible to have Q/A bot being able to answer most of the questions about UFC.

Back then we realized that Google search gives unexpectedly good results, but next step was Semantic search with embeddings.

## Architecture

Plan was to use:
- **OpenAI** to generate embeddings and completions
- **Pinecone** to store embeddings and manage similarity search
- **MySQL** to store the data (this is optional and data can be stored in Pinecone too)
- **Langchain** to orchestrate communication between all above

We used Langchain's CombineDocuments chains. There are 4 types of them, explained [HERE](https://langchain.readthedocs.io/en/latest/modules/chains/combine_docs.html){:target="_blank"}{:rel="noopener noreferrer"} and we tested all of them.

## Getting data

As mentioned in previous post we decided to use data from Wikipedia. Here in big part we followed [OpenAI cookbook](https://github.com/openai/openai-cookbook/){:target="_blank"}{:rel="noopener noreferrer"} to download all we needed. We quickly realized that most interesting data within UFC pages is stored in tables, which are totally ignored by Wikipedia python package :( So we needed to adjust the script accordingly.

Finally we pulled the data from Wikipedia, put it into MySQL. Then generated embeddings and stored them in  Pinecone. Each document in Pinecone consists (through metadata) of database id, which will later allow us to extract proper data from MySQL.

Example metadata:

```
{
  "dbId": "610",
  "lastUpdateDate": "2023-02-04",
  "title": ""List of UFC champions"",
  "tokens": 188
}
```

Quick diagram

![Expression]({{ "/assets/images/Post2.TechStack.png" | relative_url }})

## Code

First some code for documentation purposes. I do not add part with feeding Pinecone, because there are many good tutorials how to do it, for example [HERE](https://docs.pinecone.io/docs/openai){:target="_blank"}{:rel="noopener noreferrer"}

### Common part

This is common part for all chains type. Here we:
- initialize Pinecone/OpenAI services,
- we search existing Pinecone documents for similarity with user question
- extract database ID from metadata of those documents
- extract data from database

{% highlight python %}
import pinecone
from langchain.vectorstores import Pinecone
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain import OpenAI
from langchain.chains.question_answering import load_qa_chain
from langchain.prompts.base import RegexParser
from langchain.prompts import PromptTemplate
from langchain.docstore.document import Document
from contentDBApiMethods import getContentFromDB

embeddings = OpenAIEmbeddings()

OPENAI_API_KEY=config.defaults['openai_api_key']
PINECONE_API_KEY=config.defaults['pinecone_api_key']

# initialize pinecone / llm
pinecone.init(api_key=PINECONE_API_KEY, environment="us-west1-gcp")
index_name = "ufc"
index = pinecone.Index(index_name)
llm = OpenAI(temperature=0,model_name='text-davinci-003')

# using already existing index
docsearch = Pinecone.from_existing_index(index_name, embeddings, text_key="dbId")

# first use similarity search to get top 3 results
docs = docsearch.similarity_search(query, 3)

# then get ID of DB entry for each result
finalDocs=[]
for doc in docs:
  dbId=doc.page_content
  # function to get DB content
  dbData=getContentFromDB(dbId)
  dbDataFinal = dbData['Message']['data'][0]['text']
  # tabular data - is alternative method of extracting tables from wiki (python list)
  dbDataTabularPerId = dbData['Message']['data'][0]['textTabular']
  # if tabular data exists in db - use it
  if dbDataTabularPerId != '':
    dbDataFinal = dbDataTabularPerId

  finalDocs.append(Document(page_content=dbDataFinal, metadata=doc.metadata))
{% endhighlight %}

Having that - we can go to next step. Using Langchain's CombineDocuments chains get the best possible answer for user's question. Here is code for each of the method (using custom prompts).

### Stuffing

This method is simply putting together all extracted documents as context for LLM.

{% highlight python %}
  template = """Use the following portion of a long document to see if any of the text is relevant to answer the question.
  {context}
  Question: {query}
  Relevant text, if any:"""
  PROMPT = PromptTemplate(template=template, input_variables=["context", "query"])

  chain = load_qa_chain(llm, chain_type="stuff", verbose=True, prompt=PROMPT)

  result = chain.run(input_documents=finalDocs, query=query)
{% endhighlight %}

### Map reduce

This method requires more queries to LLM as first individual chunks are used as context and then everything is combined together.

{% highlight python %}
  # query for individual chunks
  question_prompt_template = """Use the following portion of a long document to see if any of the text is relevant to answer the question.
  {context}
  Question: {question}
  Relevant text, if any:"""
  QUESTION_PROMPT = PromptTemplate(
      template=question_prompt_template, input_variables=["context", "question"]
  )

  # query for final step - combine all to generate final answer
  combine_prompt_template = """Given the following extracted parts of a long document and a question, create a final answer.
  If you don't know the answer, just say that you don't know. Don't try to make up an answer.

  QUESTION: {question}
  =========
  {summaries}
  =========
  FINAL ANSWER:"""
  COMBINE_PROMPT = PromptTemplate(
      template=combine_prompt_template, input_variables=["summaries", "question"]
  )
  chain = load_qa_chain(llm, chain_type="map_reduce", return_intermediate_steps=True,   question_prompt=QUESTION_PROMPT, combine_prompt=COMBINE_PROMPT)
  result = chain({"input_documents": finalDocs, "question": query}, return_only_outputs=False)
{% endhighlight %}

### Map-rerank

Similar to map reduce - but gives a score for each chunks of context - and highest score is considered as best answer.

{% highlight python %}
  # we have to parse the output - because it consists of 2 parts (answer itself and its score)
  output_parser = RegexParser(
      regex=r"(\S.*)\s*Score: (.*)",
      output_keys=["answer", "score"],
      default_output_key = "score"
  )

  prompt_template = """Use the following pieces of context to answer the question at the end. If the answer is not contained within the context below, say 'I don't know'. If you don't know the answer, just say that you don't know, don't guess and don't try to make up an answer.

  In addition to giving an answer, also return a score of how fully it answered the user's question. This should be in the following format:

  Question: [question here]
  Helpful Answer: [answer here]
  Score: [score between 0 and 100]

  Begin!

  Context:
  ---------
  {context}
  ---------
  Question: {question}
  Helpful Answer:"""
  PROMPT = PromptTemplate(
      template=prompt_template,
      input_variables=["context", "question"],
      output_parser=output_parser,
  )

  chain = load_qa_chain(llm, chain_type="map_rerank", return_intermediate_steps=True, prompt=PROMPT)
  result = chain({"input_documents": finalDocs, "question": query}, return_only_outputs=False)
{% endhighlight %}

### Refine

This uses each first chunk to generate the output and then all others are added in separate steps to refine generated answer. This obviously cannot be parallelized. For our UFC bot it was giving worst results from very beginning, so we quickly dropped it.

{% highlight python %}
  # prompt used to refine existing answer
  refine_template = (
    "The original question is as follows: {question}\n"
    "We have provided an existing answer, including sources: {existing_answer}\n"
    "We have the opportunity to refine the existing answer"
    "(only if needed) with some more context below.\n"
    "------------\n"
    "{context_str}\n"
    "------------\n"
    "Given the new context, refine the original answer to better "
    "answer the question"
    "If the context isn't useful, return the original answer."
  )
  refine_prompt = PromptTemplate(
      input_variables=["question", "existing_answer", "context_str"],
      template=refine_template,
  )

  # initial prompt (for 1st chunk)
  question_template = (
    "Context information is below. \n"
    "---------------------\n"
    "{context_str}"
    "\n---------------------\n"
    "Given the context information and not prior knowledge, "
    "answer the question: {question}\n"
  )
  question_prompt = PromptTemplate(
      input_variables=["context_str", "question"], template=question_template
  )

  chain = load_qa_chain(llm, chain_type="refine", return_intermediate_steps=True, question_prompt=question_prompt, refine_prompt=refine_prompt)
  result = chain({"input_documents": finalDocs, "question": query}, return_only_outputs=False)
{% endhighlight %}

## Summary

In fact this few steps should be enough to build any QA bot. You just need to provide it with proper data. It's amazing how easy it is these days.

LLMs can significantly reduce complexity of your code, but with the addition of all those supportive tools, the ease of use is taken to another level. So excited about this technology!
