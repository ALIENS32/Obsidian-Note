The input to a multimodal model is denoted as $I$, with textual input specified as $I_t$ and visual input as $I_v$. To evaluate the input scrutiny capability of multimodal large models, we construct erroneous inputs $e$ by rewriting $I_t$ and implanting seven types of predefined errors into it separately.

Notably, for certain error types, inconsistencies may arise between visual and textual inputs when $\exists c \in \text{Sem}(I_v), c' \in \text{Sem}(I_t)$ where $c \perp c'$. This condition indicates that at least one semantic concept $c$ from the visual input logically contradicts a semantic concept $c'$ from the textual input. Such cases are categorized as \textit{Cross-Modal Inconsistency}, a specific error type characterized by direct conflicts in semantic or factual information between $I_v$ and $I_t$.

A model is considered to have input scrutiny capability under current input $I$ when its output $A$ identifies the implanted error $e$ without relying on explicit prompting to "check for errors".

