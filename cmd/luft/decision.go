package main

import (
	"fmt"
	"strings"

	"github.com/lukemuz/luft"
	"github.com/lukemuz/luft/tools/decision"
)

func configureDecision(model, endpoint, key string, disabled bool) (luft.Toolset, error) {
	if disabled || model == "" {
		return luft.Toolset{}, nil
	}
	if strings.TrimSpace(key) == "" {
		return luft.Toolset{}, fmt.Errorf("decision: LUFT_DECISION_API_KEY is required when -decision-model is set")
	}
	binding, err := decision.New(decision.Config{Model: model, Endpoint: endpoint, APIKey: key})
	if err != nil {
		return luft.Toolset{}, err
	}
	return luft.Tools(binding), nil
}
