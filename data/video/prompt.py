class PromptBuilder:
    base_prompt = (
            "Generate a high-fidelity 360-degree panoramic video of the object shown in the provided reference images.\n"
            "Requirements:\n"
            "1. The object must be fully visible in every frame; no parts should be cropped or cut off.\n"
            "2. Maintain consistent scale, orientation, and position of the object across all frames.\n"
            "3. Ensure smooth rotation; no sudden jumps, distortions, or unnatural movements.\n"
            "4. The background should be minimal, neutral, or lightly styled, emphasizing the object.\n"
            "5. Ensure consistent lighting, perspective, and color tone throughout.\n"
            "6. Produce frames of uniform high visual fidelity, suitable for downstream training datasets.\n"
            "7. Avoid any sudden changes or anomalies between consecutive frames.\n"
            "Duration: 5 seconds."
        )

    @classmethod
    def build_prompt(cls, additional_instructions: str = None) -> str:
        if additional_instructions:
            return f"{cls.base_prompt}\n{additional_instructions}"
        return cls.base_prompt