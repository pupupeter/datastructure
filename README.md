# datastructure


[한국어](https://github.com/pupupeter/datastructure/blob/main/readmekorean.md)

# project proposal  

Midterm:https://youtu.be/2sRbDFhY628

Midterm2:https://youtu.be/AMGhJTg5nto

#### 因為程式碼可能秀的不是很清楚，所以給大家連結https://github.com/pupupeter/datastructure/blob/main/%E6%9C%80%E6%96%B0%E7%9A%84%E9%80%B2%E5%BA%A60413.py

Final project:

https://youtu.be/v94G0bUrNU4



# AI Professor Debate System (Based on AutoGen/SmolEAgent)

## 1. Introduction
This system utilizes AutoGen and SmolEAgent to implement AI professor debates, incorporating PDF RAG (Retrieval-Augmented Generation) and Web search to enhance knowledge sources, ensuring efficiency and accuracy in the debate process.

## 2. Key Components

### 2.1 SmolEAgent
- **Lightweight AI Agent**: Suitable for embedded environments and works collaboratively with AutoGen.
- **Modular Design**: Allows for adding different reasoning capabilities as needed.

### 2.2 PDF RAG
- **Document Retrieval**: Enables AI agents to query relevant content from academic PDFs.
- **Context Enhancement**: Uses retrieved content to supplement AI-generated responses.

### 2.3 Web Search
- **Real-Time Information Retrieval**: Ensures that AI agents base their arguments on the latest knowledge.
- **Result Filtering**: Avoids misinformation or inaccurate data affecting the debate.

## 3. Workflow
![image](https://github.com/user-attachments/assets/96b91b3d-eb4c-4a24-908c-a8d8cc478124)

1. **Input Debate Topic**
   - The user or system defines the debate topic.
2. **Retrieve Relevant Data**
   - PDF RAG searches academic literature.
   - Web search fetches the latest related information.
3. **AI Role Assignment**
   - AI Professor A: Supports the argument.
   - AI Professor B: Opposes the argument.
   - AI Judge: Evaluates the validity of both arguments or stop the action.
4. **Debate Process**
   - AI professors take turns presenting their points.
   - Arguments can be dynamically adjusted based on new information.
5. **Final Evaluation**
   - The AI judge assesses the content quality and persuasiveness of the arguments.
   
## 4. Potential Applications
- **Academic Discussions**
- **Legal Debate Simulations**
- **Policy Analysis**
- **Student AI Interaction for Debate Training**




## 5. Future Expansion
- **Multilingual Support**
- **Integration of Knowledge Graphs for Better Reasoning**
- **Integration with Academic Databases (e.g., Arxiv, Google Scholar)"






