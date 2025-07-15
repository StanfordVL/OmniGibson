
fwd_prompt = """
You are a capable agent designed to infer forward dynamics transitions in embodied decision-making. Analyze the provided images:
- The first image represents the current state of the scene.
- The subsequent four images (A, B, C, D) represent potential next states based on the given state changes.

Your task is to identify which choice (A, B, C, or D) is the most likely next state after the following state changes take place:

{STATE_CHANGES}

Focus on the visual details in the images, specifically observing the positions, placements, and relationships of the objects mentioned.

Do not perform any reasoning or explanation; simply select the letter corresponding to the most likely next state.
"""

inv_prompt = """You are a capable agent designed to infer inverse dynamics transitions in embodied decision-making. Analyze the provided images:
- The first image represents the **current state** of the scene.
- The second image represents the **next state** of the scene.

Your task is to determine which set of state changes listed below(A, B, C, or D) most accurately describes the action(s) that occurred between the two states. Each choice contains a list of state changes that describe object interactions and positional changes:


{STATE_CHANGES_CHOICES}
Focus on the visual details in the images, such as the relationships, placements, and interactions of the objects mentioned in the choices. Use this information to identify the correct sequence of state changes. Do not perform any reasoning or explanation; simply select the letter corresponding to the correct set of state changes, and only generate a single capitalized letter as your output.
"""