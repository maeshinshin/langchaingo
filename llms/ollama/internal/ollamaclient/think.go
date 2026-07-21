package ollamaclient

import (
	"encoding/json"
	"strconv"
)

// ThinkValue represents the Ollama "think" field, which the server accepts
// either as a JSON boolean (true / false) or as a thinking-level string
// ("high", "medium", "low", "max"). Use [NewBoolThink] or [NewStringThink]
// to construct one, or leave it nil to omit the field entirely.
type ThinkValue struct {
	bool   *bool
	string *string
}

// NewBoolThink builds a ThinkValue from a boolean.
func NewBoolThink(v bool) *ThinkValue {
	return &ThinkValue{bool: &v}
}

// NewStringThink builds a ThinkValue from a thinking-level string.
func NewStringThink(v string) *ThinkValue {
	return &ThinkValue{string: &v}
}

// MarshalJSON renders ThinkValue as either a JSON boolean or a JSON string,
// depending on which variant was provided.
func (t ThinkValue) MarshalJSON() ([]byte, error) {
	switch {
	case t.bool != nil:
		return strconv.AppendBool(nil, *t.bool), nil
	case t.string != nil:
		return json.Marshal(*t.string)
	default:
		return []byte("null"), nil
	}
}

// IsEnabled reports whether the ThinkValue represents an enabled thinking
// request. A nil ThinkValue, an explicit false, or the string "false" all
// count as disabled.
func (t *ThinkValue) IsEnabled() bool {
	if t == nil {
		return false
	}
	if t.bool != nil {
		return *t.bool
	}
	if t.string != nil {
		return *t.string != "" && *t.string != "false"
	}
	return false
}
