"""Tests for the bearer-token settings (issue #15): pure parsing, no app.

Every token in this file is synthetic.
"""

import pytest

from calendar_agent import auth

TOKEN_A = "synthetic-token-alpha-0123456789abcdef"
TOKEN_B = "synthetic-token-bravo-0123456789abcdef"


class TestParseApiKeys:
    def test_one_pair_yields_one_named_key(self):
        keys = auth.parse_api_keys(f"laptop:{TOKEN_A}")
        assert [key.name for key in keys] == ["laptop"]

    def test_pairs_are_comma_separated_and_keep_their_order(self):
        keys = auth.parse_api_keys(f"laptop:{TOKEN_A},scheduler:{TOKEN_B}")
        assert [key.name for key in keys] == ["laptop", "scheduler"]

    def test_whitespace_around_entries_names_and_tokens_is_ignored(self):
        keys = auth.parse_api_keys(f"  laptop : {TOKEN_A} ,\n scheduler:{TOKEN_B}  ")
        assert [key.name for key in keys] == ["laptop", "scheduler"]
        assert auth.match_token(keys, TOKEN_A.encode()) == "laptop"

    @pytest.mark.parametrize("raw", [None, "", "   ", ",", " , ,"])
    def test_unset_or_blank_is_no_keys(self, raw):
        assert auth.parse_api_keys(raw) == ()

    def test_a_trailing_comma_is_tolerated(self):
        keys = auth.parse_api_keys(f"laptop:{TOKEN_A},")
        assert [key.name for key in keys] == ["laptop"]

    def test_a_key_does_not_keep_the_token_text(self):
        """What is stored is a digest, so a stray repr() of the settings (a
        debugger, a log line someone adds later) cannot print a token."""
        keys = auth.parse_api_keys(f"laptop:{TOKEN_A}")
        assert TOKEN_A not in repr(keys)


class TestMatchToken:
    def test_the_right_token_names_its_caller(self):
        keys = auth.parse_api_keys(f"laptop:{TOKEN_A},scheduler:{TOKEN_B}")
        assert auth.match_token(keys, TOKEN_A.encode()) == "laptop"
        assert auth.match_token(keys, TOKEN_B.encode()) == "scheduler"

    @pytest.mark.parametrize(
        "presented",
        [
            b"",
            b"synthetic-token-wrong-0123456789abcdef",
            TOKEN_A.encode()[:-1],
            TOKEN_A.encode() + b"x",
            TOKEN_A.upper().encode(),
        ],
    )
    def test_anything_else_matches_nobody(self, presented):
        keys = auth.parse_api_keys(f"laptop:{TOKEN_A}")
        assert auth.match_token(keys, presented) is None

    def test_no_keys_match_nobody(self):
        assert auth.match_token((), TOKEN_A.encode()) is None

    def test_every_configured_key_is_compared_with_hmac_compare_digest(self, monkeypatch):
        """The comparison goes through ``hmac.compare_digest``, and a match
        does not end the loop early: how long a check takes does not depend
        on which key matched or whether one did."""
        import hmac

        calls = []

        def spy(a, b):
            calls.append((a, b))
            return hmac.compare_digest(a, b)

        monkeypatch.setattr(auth, "_constant_time_equal", spy)
        keys = auth.parse_api_keys(f"laptop:{TOKEN_A},scheduler:{TOKEN_B}")

        assert auth.match_token(keys, TOKEN_A.encode()) == "laptop"
        assert len(calls) == 2, "stopped comparing after the first match"
        calls.clear()
        assert auth.match_token(keys, b"synthetic-token-wrong-0123456789abcdef") is None
        assert len(calls) == 2

    def test_the_comparison_seam_is_hmac_compare_digest(self):
        import hmac

        assert auth._constant_time_equal is hmac.compare_digest


SHORT_TOKEN = "synthetic-short-token"  # 21 characters
LONG_NAME = "n" * 32  # one past the longest name
MALFORMED = {
    "no colon at all": f"laptop{TOKEN_A}",
    "no name": f":{TOKEN_A}",
    "blank name": f"   :{TOKEN_A}",
    "empty token": "laptop:",
    "blank token": "laptop:    ",
    "token below the minimum length": f"laptop:{SHORT_TOKEN}",
    "duplicate name": f"laptop:{TOKEN_A},laptop:{TOKEN_B}",
    "duplicate token": f"laptop:{TOKEN_A},scheduler:{TOKEN_A}",
    "name with a space": f"my laptop:{TOKEN_A}",
    "name with a slash": f"lap/top:{TOKEN_A}",
    "token in quotes": f'laptop:"{TOKEN_A}"',
    "token with a space inside": f"laptop:{TOKEN_A[:20]} {TOKEN_A[20:]}",
    "token with a non-ascii character": f"laptop:{TOKEN_A}é",
    "second entry bad, first fine": f"laptop:{TOKEN_A},:{TOKEN_B}",
    "name and token swapped": f"{TOKEN_A}:laptop",
    "a token in the name slot": f"{TOKEN_A}:{TOKEN_B}",
    "the same token in both slots": f"{TOKEN_A}:{TOKEN_A}",
    "name longer than 31 characters": f"{LONG_NAME}:{TOKEN_A}",
}


class TestMalformedApiKeys:
    @pytest.mark.parametrize("case", sorted(MALFORMED))
    def test_malformed_value_is_refused(self, case):
        with pytest.raises(auth.AuthConfigError):
            auth.parse_api_keys(MALFORMED[case])

    @pytest.mark.parametrize("case", sorted(MALFORMED))
    def test_the_refusal_names_the_setting_and_echoes_nothing_from_it(self, case):
        """The message points at the setting and the entry's position. It
        repeats no part of the value: a token pasted in the wrong place (the
        name slot, an entry with no colon) must not reach a log through the
        error that reports it."""
        raw = MALFORMED[case]
        with pytest.raises(auth.AuthConfigError) as excinfo:
            auth.parse_api_keys(raw)
        message = str(excinfo.value)
        assert "CALENDAR_AGENT_API_KEYS" in message
        for token in (TOKEN_A, TOKEN_B, SHORT_TOKEN, TOKEN_A[:20], TOKEN_A[20:]):
            assert token not in message
        for name in ("laptop", "scheduler", "lap/top", "my laptop", LONG_NAME):
            assert name not in message

    def test_the_refusal_says_which_entry(self):
        with pytest.raises(auth.AuthConfigError, match="entry 2"):
            auth.parse_api_keys(f"laptop:{TOKEN_A},:{TOKEN_B}")

    def test_a_name_is_1_to_31_characters(self):
        assert len(auth.parse_api_keys("n" * 31 + f":{TOKEN_A}")) == 1
        assert len(auth.parse_api_keys(f"n:{TOKEN_A}")) == 1
        with pytest.raises(auth.AuthConfigError, match="31"):
            auth.parse_api_keys("n" * 32 + f":{TOKEN_A}")

    def test_no_accepted_token_is_short_enough_to_be_a_name(self):
        """A name is at most one character shorter than the shortest token,
        so a token pasted into the name slot is refused rather than accepted
        as a name and then printed in every log line that names the caller."""
        assert auth.MAX_NAME_LENGTH == auth.MIN_TOKEN_LENGTH - 1
        with pytest.raises(auth.AuthConfigError, match="entry 1"):
            auth.parse_api_keys("a" * auth.MIN_TOKEN_LENGTH + f":{TOKEN_A}")

    def test_minimum_token_length_is_32(self):
        assert auth.MIN_TOKEN_LENGTH == 32
        assert len(auth.parse_api_keys("laptop:" + "a" * 32)) == 1
        with pytest.raises(auth.AuthConfigError, match="32"):
            auth.parse_api_keys("laptop:" + "a" * 31)


class TestResolveAuthMode:
    @pytest.mark.parametrize("raw", [None, "", "   "])
    def test_unset_or_blank_is_required(self, raw):
        assert auth.resolve_auth_mode(raw) is auth.AuthMode.REQUIRED

    @pytest.mark.parametrize("raw", ["required", "optional", "off"])
    def test_the_three_modes(self, raw):
        assert auth.resolve_auth_mode(raw).value == raw
        assert auth.resolve_auth_mode(f"  {raw.upper()} ").value == raw
        assert {mode.value for mode in auth.AuthMode} == {"required", "optional", "off"}

    @pytest.mark.parametrize(
        "raw", ["on", "true", "none", "disabled", "require", "0", "off,required"]
    )
    def test_anything_else_is_refused_without_being_echoed(self, raw):
        """An unknown value is not read as "off" (or as anything): the service
        does not start. The value is not repeated, in case a token was pasted
        into the wrong setting."""
        with pytest.raises(auth.AuthConfigError) as excinfo:
            auth.resolve_auth_mode(raw)
        message = str(excinfo.value)
        assert "CALENDAR_AGENT_AUTH" in message
        for allowed in ("required", "optional", "off"):
            assert allowed in message
        assert f"'{raw}'" not in message and f'"{raw}"' not in message


QUOTE_HINT = "check for quotation marks around the value"


class TestQuotedValues:
    """``docker run --env-file`` keeps quotation marks as part of the value,
    so a correctly typed value in quotes is refused. The message says to
    check for them, and still repeats nothing from the value."""

    @pytest.mark.parametrize("quote", ['"', "'"])
    def test_a_quoted_mode_is_refused_with_the_hint(self, quote):
        raw = f"{quote}required{quote}"
        with pytest.raises(auth.AuthConfigError) as excinfo:
            auth.load_auth_settings({auth.MODE_ENV: raw, auth.KEYS_ENV: f"laptop:{TOKEN_A}"})
        message = str(excinfo.value)
        assert QUOTE_HINT in message
        assert raw not in message
        assert TOKEN_A not in message and "laptop" not in message

    @pytest.mark.parametrize("quote", ['"', "'"])
    def test_a_quoted_keys_value_is_refused_with_the_hint(self, quote):
        raw = f"{quote}laptop:{TOKEN_A}{quote}"
        with pytest.raises(auth.AuthConfigError) as excinfo:
            auth.load_auth_settings({auth.KEYS_ENV: raw})
        message = str(excinfo.value)
        assert QUOTE_HINT in message
        assert "entry 1" in message
        for fragment in (TOKEN_A, TOKEN_A[:16], TOKEN_A[-16:], "laptop"):
            assert fragment not in message


class TestLoadAuthSettings:
    def test_required_with_tokens(self):
        settings = auth.load_auth_settings({"CALENDAR_AGENT_API_KEYS": f"laptop:{TOKEN_A}"})
        assert settings.mode is auth.AuthMode.REQUIRED
        assert [key.name for key in settings.keys] == ["laptop"]

    @pytest.mark.parametrize(
        "environ",
        [
            {},
            {"CALENDAR_AGENT_AUTH": "required"},
            {"CALENDAR_AGENT_AUTH": "required", "CALENDAR_AGENT_API_KEYS": ""},
            {"CALENDAR_AGENT_API_KEYS": " , "},
        ],
    )
    def test_required_with_no_tokens_is_refused_naming_both_settings(self, environ):
        with pytest.raises(auth.AuthConfigError) as excinfo:
            auth.load_auth_settings(environ)
        message = str(excinfo.value)
        assert "CALENDAR_AGENT_API_KEYS" in message
        assert "CALENDAR_AGENT_AUTH" in message

    def test_optional_with_tokens(self):
        settings = auth.load_auth_settings(
            {"CALENDAR_AGENT_AUTH": "optional", "CALENDAR_AGENT_API_KEYS": f"laptop:{TOKEN_A}"}
        )
        assert settings.mode is auth.AuthMode.OPTIONAL
        assert [key.name for key in settings.keys] == ["laptop"]

    def test_optional_with_no_tokens_is_allowed(self):
        settings = auth.load_auth_settings({"CALENDAR_AGENT_AUTH": "optional"})
        assert settings.mode is auth.AuthMode.OPTIONAL
        assert settings.keys == ()

    @pytest.mark.parametrize("mode", ["required", "optional"])
    def test_a_malformed_keys_value_is_refused_in_required_and_optional(self, mode):
        with pytest.raises(auth.AuthConfigError):
            auth.load_auth_settings(
                {"CALENDAR_AGENT_AUTH": mode, "CALENDAR_AGENT_API_KEYS": f"laptop:{SHORT_TOKEN}"}
            )

    @pytest.mark.parametrize(
        "keys", [None, "", f"laptop:{TOKEN_A}", f"laptop:{SHORT_TOKEN}", "nonsense"]
    )
    def test_off_does_not_read_the_keys(self, keys):
        """``off`` is the undo switch, so it does not depend on the other
        setting being well-formed."""
        environ = {"CALENDAR_AGENT_AUTH": "off"}
        if keys is not None:
            environ["CALENDAR_AGENT_API_KEYS"] = keys
        settings = auth.load_auth_settings(environ)
        assert settings.mode is auth.AuthMode.OFF
        assert settings.keys == ()

    def test_an_unknown_mode_is_refused_even_with_good_tokens(self):
        with pytest.raises(auth.AuthConfigError):
            auth.load_auth_settings(
                {"CALENDAR_AGENT_AUTH": "disabled", "CALENDAR_AGENT_API_KEYS": f"laptop:{TOKEN_A}"}
            )
