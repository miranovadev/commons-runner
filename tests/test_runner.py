"""No Ollama or Commons server required: contract-shape regressions."""
import unittest

from jsonschema import Draft202012Validator

from runner import RunnerError, prompt_for, transport_schema, validation_messages


class TransportSchemaTests(unittest.TestCase):
    def setUp(self):
        self.schema = {
            "$defs": {
                "instrument": {
                    "anyOf": [
                        {"type": "object", "additionalProperties": False,
                         "properties": {"kind": {"const": "citation"},
                                        "reference": {"type": "string", "pattern": "^a"}},
                         "required": ["kind", "reference"]},
                        {"type": "object", "additionalProperties": False,
                         "properties": {"kind": {"const": "web_search"},
                                        "query": {"type": "string"}},
                         "required": ["kind", "query"]},
                    ],
                },
                "actions": {
                    "attestation.create": {
                        "type": "object", "additionalProperties": False,
                        "required": ["action", "proposition_id", "direction"],
                        "properties": {
                            "action": {"const": "attestation.create"},
                            "proposition_id": {"type": "string", "enum": ["A", "B"]},
                            "direction": {"type": "string", "enum": ["refute", "support"]},
                        },
                        "anyOf": [
                            {"properties": {"proposition_id": {"enum": ["A"]},
                                            "direction": {"enum": ["support"]}}},
                            {"properties": {"proposition_id": {"enum": ["B"]},
                                            "direction": {"enum": ["refute"]}}},
                        ],
                    },
                    "instrument.execute": {
                        "type": "object", "additionalProperties": False,
                        "required": ["action", "proposition_id", "instrument"],
                        "properties": {
                            "action": {"const": "instrument.execute"},
                            "proposition_id": {"enum": ["A"]},
                            "instrument": {"$ref": "#/$defs/instrument"},
                        },
                        "not": {"properties": {"proposition_id": {"const": "A"}}},
                    },
                },
            },
        }

    def test_transport_preserves_action_literals_and_target_directions(self):
        result = transport_schema(self.schema)
        validator = Draft202012Validator(result)
        self.assertFalse(list(validator.iter_errors(
            {"action": "attestation.create", "proposition_id": "A",
             "direction": "support"})))
        self.assertTrue(list(validator.iter_errors(
            {"action": {"$ref": "attestation.create"}, "proposition_id": "A",
             "direction": "support"})))
        self.assertTrue(list(validator.iter_errors(
            {"action": "attestation.create", "proposition_id": "B",
             "direction": "support"})))

    def test_nested_instrument_refs_inline_and_no_obsolete_plan(self):
        result = transport_schema(self.schema)
        self.assertFalse(list(Draft202012Validator(result).iter_errors({
            "action": "instrument.execute", "proposition_id": "A",
            "instrument": {"kind": "citation", "reference": "a real identifier"},
        })))
        self.assertTrue(list(Draft202012Validator(result).iter_errors({
            "action": "instrument.execute", "proposition_id": "A",
            "instrument": {"kind": "web_search", "reference": "a wrong field"},
        })))
        self.assertNotIn("$ref", str(result))
        self.assertNotIn("'not':", str(result))

    def test_full_commons_schema_remains_authoritative(self):
        contribution = {"action": "instrument.execute", "proposition_id": "A",
                        "instrument": {"kind": "citation", "reference": "wrong"}}
        # The output transport intentionally omits the regex, but Commons does not.
        self.assertFalse(list(Draft202012Validator(transport_schema(self.schema)).iter_errors(contribution)))
        original = {"$defs": self.schema["$defs"], "oneOf": [
            {"$ref": "#/$defs/actions/attestation.create"},
            {"$ref": "#/$defs/actions/instrument.execute"},
        ]}
        self.assertTrue(validation_messages(original, contribution))

    def test_empty_or_invalid_action_inventory_fails_closed(self):
        with self.assertRaises(RunnerError):
            transport_schema({"$defs": {"actions": {}}})

    def test_prompt_consumes_cognitive_context_not_operational_plan(self):
        state = {"context": {
            "participant": {"callsign": "Qwen"},
            "read": {"propositions": []},
            "gate": {"acting_id": "A", "propositions": []},
            "write": {"question": "What is different?", "propositions": []},
            "receipts": [],
        }}
        messages = prompt_for(state)
        self.assertEqual(__import__("json").loads(messages[1]["content"]), state["context"])
        self.assertNotIn("plan", messages[1]["content"])


if __name__ == "__main__":
    unittest.main()
