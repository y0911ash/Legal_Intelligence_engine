import os
import unittest


class TestUI(unittest.TestCase):
    def test_style_css_exists(self):
        self.assertTrue(os.path.exists("style.css"), "style.css file should exist")

    def test_style_css_uses_modern_design_tokens(self):
        with open("style.css", "r", encoding="utf-8") as file:
            content = file.read()
            self.assertIn(":root {", content, "style.css should define CSS design tokens")
            self.assertIn("Instrument+Serif", content, "style.css should load the modern serif accent font")
            self.assertIn(".modern-hero", content, "style.css should style the new hero section")

    def test_app_has_modern_layout_sections(self):
        with open("app.py", "r", encoding="utf-8") as file:
            content = file.read()
            self.assertIn("modern-hero", content, "app.py should render the hero section")
            self.assertIn("Pipeline progress", content, "app.py should render the pipeline progress section")
            self.assertIn("Load a judgment", content, "app.py should render the input workspace section")
            self.assertIn("Run the legal intelligence pipeline", content, "app.py should render the analysis workspace section")
            self.assertIn("Review the intelligence report", content, "app.py should render the results workspace section")

    def test_app_retains_session_state_flow(self):
        with open("app.py", "r", encoding="utf-8") as file:
            content = file.read()
            self.assertIn("st.session_state.step", content, "app.py should continue using session_state flow")
            self.assertIn("st.session_state.pipeline_result", content, "app.py should store the pipeline result")
            self.assertIn("Start New Case Analysis", content, "app.py should allow resetting the case flow")

    def test_app_has_exports_and_comparison_view(self):
        with open("app.py", "r", encoding="utf-8") as file:
            content = file.read()
            self.assertIn("st.download_button(", content, "app.py should provide download actions")
            self.assertIn("Download JSON Report", content, "app.py should expose JSON export")
            self.assertIn("Download Case Brief (.txt)", content, "app.py should expose text report export")
            self.assertIn("Side-by-side review", content, "app.py should expose a comparison view")
            self.assertIn("source_judgment_view", content, "app.py should render the source judgment panel")
            self.assertIn("generated_summary_view", content, "app.py should render the generated summary panel")

    def test_app_no_legacy_sidebar_navigation_copy(self):
        with open("app.py", "r", encoding="utf-8") as file:
            content = file.read()
            self.assertNotIn("with st.sidebar", content, "app.py should no longer use the old sidebar wizard")
            self.assertNotIn("class=\"title-bar\"", content, "app.py should no longer use the old title bar")


if __name__ == "__main__":
    unittest.main()
