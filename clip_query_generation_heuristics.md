Summary of Heuristics for Generating Text Descriptions

When generating text descriptions for each object class, the following guidelines and heuristics were applied to ensure the retrieval of multiple images with similar physical characteristics suitable for 3D reconstruction using CLIP embeddings:

	1.	Focus on Physical Attributes:
	•	Specify Key Physical Qualities: Include specific physical attributes such as color, type, or distinctive features to ensure coherence in the retrieved images.
	•	Examples: “A red sports car”, “A black leather handbag”, “A golden retriever”.
	•	Avoid Actions or Behaviors: Do not describe actions, activities, or behaviors of the objects to maintain focus on their physical appearance.
	•	Avoid: “A dog playing fetch”, “A person jogging”.
	•	Use Instead: “A black Labrador”, “A person in a red tracksuit”.
	2.	General yet Specific Descriptions:
	•	Balance Specificity: Make descriptions general enough to retrieve a large number of images but specific enough to maintain similar physical characteristics.
	•	Avoid Overly Specific Contexts: Do not include detailed situations or rare scenarios that might limit image availability.
	•	Avoid: “A group of people with umbrellas during a parade”.
	•	Use Instead: “People with umbrellas”.
	3.	Conciseness and Simplicity:
	•	Short Descriptions: Keep descriptions brief and to the point, focusing on essential physical attributes.
	•	Avoid Excessive Details: Remove unnecessary specifics that do not contribute to the main physical characteristics.
	•	Change From: “A black luxury sedan with tinted windows”.
	•	Change To: “A black luxury sedan”.
	4.	Diversity within Object Classes:
	•	Variety of Descriptions: Provide different descriptions within each object class to cover various subcategories and attributes, ensuring a diverse set of images.
	•	Examples for “Cat”: “A black cat with green eyes”, “An orange tabby cat”, “A Siamese cat”.
	5.	Consideration of CLIP Embeddings:
	•	Visual Attribute Emphasis: Focus on visual features that are easily recognizable and distinguishable in images, which CLIP embeddings can effectively associate.
	•	Avoid Ambiguity: Use clear and unambiguous language to improve the accuracy of image retrieval.
	•	Avoid: Vague descriptors like “nice car”.
	•	Use Instead: “A red sports car”.
	6.	Consistency in Structure:
	•	Uniform Formatting: Maintain a consistent sentence structure across all descriptions for clarity and ease of use.
	•	Structure: Begin with an indefinite article (“A” or “An”), followed by the object with its attributes.
	•	Example: “A [attribute] [object]”.
	7.	Formatting for LLM Use:
	•	JSON Style Organization: Format the descriptions in a JSON structure where each object class is a key, and the value is a list of text descriptions.
	•	Purpose: Facilitates easy parsing and utilization by language models for further text generation.
	8.	Avoiding Contextual Limitations:
	•	Exclude Temporal or Cultural Specifics: Do not include time-specific or culturally unique elements that may not be universally represented in the dataset.
	•	Avoid: “A Christmas tree with ornaments”, “A person celebrating Holi festival”.
	•	Use Instead: “An evergreen tree”, “A person covered in colorful powder”.
	9.	No Disallowed Content:
	•	Adherence to Guidelines: Ensure all descriptions comply with content policies, avoiding any disallowed or sensitive content.
	10.	Examples for Reference:
	•	Provide sample descriptions following these heuristics for each object class to illustrate the application of the guidelines.
	•	Example for “Bicycle”:
	•	“A red mountain bike”
	•	“A vintage bike with basket”
	•	“A yellow road bike”

Purpose of These Heuristics

	•	Enhance Image Retrieval: By focusing on physical attributes and avoiding actions or overly specific contexts, the descriptions are more likely to retrieve a larger set of relevant images.
	•	Facilitate 3D Reconstruction: Coherent physical characteristics in the retrieved images are essential for successful 3D reconstruction.
	•	Optimize for CLIP Embeddings: Tailoring descriptions to the strengths of CLIP embeddings improves the accuracy and relevance of image-text associations.

Using This Summary for LLM Prompting

	•	Guidance for Generation: This summary serves as a guideline for language models to generate additional text descriptions that align with the desired criteria.
	•	Ensuring Consistency: By following the outlined heuristics, LLMs can produce descriptions that are consistent in style and content, aiding in data augmentation or expansion tasks.

Example Prompt for LLMs Based on These Heuristics

	Generate 10 short and concise text descriptions for each of the following object classes. Focus on specifying key physical attributes such as color, type, or distinctive features without mentioning actions or behaviors. Ensure the descriptions are general enough to retrieve multiple images but specific enough to maintain similar physical characteristics. Format the descriptions in a consistent structure, starting with “A” or “An”, followed by the attribute and object (e.g., “A red sports car”). Organize the results in a JSON format where each object class is a key, and the value is a list of descriptions.