"""No Ollama or Commons server required: contract-shape regressions."""
import unittest

from jsonschema import Draft202012Validator

from runner import (RunnerError, prompt_for, transport_schema, validation_messages,
                    prohibited_instrument_invocations, prohibited_instrument_match)


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

    def test_instrument_exclusions_are_explained_without_changing_generation_schema(self):
        schema = {
            **self.schema,
            "x-commons": {"instrument_exclusions": {
                "A": [{"receipt_id": "R1", "invocation": {
                    "kind": "citation", "reference": "A",
                }}],
            }},
        }
        denied = {"action": "instrument.execute", "proposition_id": "A",
                  "instrument": {"kind": "citation", "reference": "A"}}
        allowed = {"action": "instrument.execute", "proposition_id": "A",
                   "instrument": {"kind": "citation", "reference": "a different"}}
        original = {"$defs": schema["$defs"], "oneOf": [
            {"$ref": "#/$defs/actions/instrument.execute"},
        ]}
        original["$defs"]["actions"]["instrument.execute"]["not"] = {
            "anyOf": [{"required": ["proposition_id", "instrument"], "properties": {
                "proposition_id": {"const": "A"},
                "instrument": {"required": ["kind", "reference"], "properties": {
                    "kind": {"const": "citation"},
                    "reference": {"enum": ["A"]},
                }},
            }}],
        }
        original["x-commons"] = schema["x-commons"]
        self.assertEqual(prohibited_instrument_invocations(schema), [{
            "proposition_id": "A", "instrument": {"kind": "citation", "reference": "A"},
        }])
        self.assertEqual(prohibited_instrument_match(schema, denied)["proposition_id"], "A")
        self.assertIsNone(prohibited_instrument_match(schema, allowed))
        self.assertIsNone(prohibited_instrument_match(schema, {"action": "participant.rest"}))
        state = {"context": {"read": {"propositions": []},
                             "gate": {"propositions": []},
                             "write": {"propositions": []},
                             "receipts": []}, "schema": original}
        messages = prompt_for(state)
        self.assertIn("PROHIBITED instrument.execute", messages[-1]["content"])
        self.assertIn('"reference":"A"', messages[-1]["content"])
        self.assertIn("already been performed or is excluded",
                      validation_messages(original, denied)[0])
        self.assertFalse(validation_messages(original, allowed))
        # Ollama's transport is intentionally permissive; Commons still rejects.
        self.assertFalse(list(Draft202012Validator(transport_schema(original)).iter_errors(denied)))

    def test_no_exclusions_means_no_extra_prompt(self):
        state = {"context": {"receipts": [], "read": {"propositions": []},
                             "gate": {"propositions": []}, "write": {"propositions": []}},
                 "schema": self.schema}
        self.assertEqual(len(prompt_for(state)), 3)
        self.assertEqual(prohibited_instrument_invocations(self.schema), [])

    def test_single_acquisition_defer_requires_actual_fields(self):
        schema = {"$defs": {"actions": {"acquisition.defer": {
            "type": "object", "additionalProperties": False,
            "required": ["action", "reason", "execution_id"],
            "properties": {
                "action": {"const": "acquisition.defer"},
                "reason": {"type": "string", "minLength": 1},
                "execution_id": {"type": "string", "const": "EXEC"},
            },
        }}}}
        output = transport_schema(schema)
        validator = Draft202012Validator(output)
        self.assertEqual(set(output["properties"]), {"action", "reason", "execution_id"})
        self.assertTrue(list(validator.iter_errors({})))
        self.assertFalse(list(validator.iter_errors(
            {"action": "acquisition.defer", "reason": "No evidence.", "execution_id": "EXEC"})))

    def test_empty_or_invalid_action_inventory_fails_closed(self):
        with self.assertRaises(RunnerError):
            transport_schema({"$defs": {"actions": {}}})

    def test_prompt_consumes_cognitive_context_not_operational_plan(self):
        state = {"schema": {"$defs": {"actions": {"participant.rest": {"required": ["action"], "properties": {"action": {"const": "participant.rest"}}}}}}, "context": {
            "participant": {"callsign": "Qwen"},
            "read": {"propositions": []},
            "gate": {"acting_id": "A", "propositions": []},
            "write": {"question": "What is different?", "propositions": []},
            "receipts": [],
        }}
        messages = prompt_for(state)
        self.assertEqual(__import__("json").loads(messages[1]["content"]), state["context"])
        self.assertNotIn("plan", messages[1]["content"])
        self.assertIn("participant.rest", messages[2]["content"])


if __name__ == "__main__":
    unittest.main()
