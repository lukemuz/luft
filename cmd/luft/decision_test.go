package main

import "testing"

func TestConfigureDecision(t *testing.T) {
	for _, tc := range []struct {
		name, model, key string
		disabled         bool
		count            int
		wantErr          bool
	}{
		{name: "unconfigured"},
		{name: "key alone", key: "k"},
		{name: "disabled", model: "m", disabled: true},
		{name: "missing key", model: "m", wantErr: true},
		{name: "enabled", model: "m", key: "k", count: 1},
	} {
		t.Run(tc.name, func(t *testing.T) {
			tools, err := configureDecision(tc.model, "", tc.key, tc.disabled)
			if (err != nil) != tc.wantErr || len(tools.Bindings) != tc.count {
				t.Fatalf("got %d tools, %v", len(tools.Bindings), err)
			}
		})
	}
}
