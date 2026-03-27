import unittest

from universal_llm import LLMConfig, LLMConfigError


class TestLLMConfig(unittest.TestCase):
    def test_vllm_defaults(self):
        config = LLMConfig.for_vllm(model="demo-model")
        resolved = config.resolve()

        self.assertEqual(resolved.provider, "vllm")
        self.assertEqual(resolved.transport, "openai_compatible")
        self.assertEqual(resolved.api_key, "EMPTY")

    def test_openai_native_transport(self):
        config = LLMConfig(provider="openai", model="gpt-4o-mini")
        resolved = config.resolve()
        self.assertEqual(resolved.transport, "native")

    def test_missing_model_raises(self):
        with self.assertRaises(LLMConfigError):
            LLMConfig(provider="openai", model="")


if __name__ == "__main__":
    unittest.main()
