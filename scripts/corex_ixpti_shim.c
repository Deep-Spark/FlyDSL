// SPDX-License-Identifier: Apache-2.0
// Shim: export ixpti* as link-time aliases to cupti* from libcupti (CoreX).

#define IXPTI_ALIAS(ix, cu) void ix(void) __attribute__((alias(#cu), visibility("default")))

IXPTI_ALIAS(ixptiActivityDisable, cuptiActivityDisable);
IXPTI_ALIAS(ixptiActivityEnable, cuptiActivityEnable);
IXPTI_ALIAS(ixptiActivityFlushAll, cuptiActivityFlushAll);
IXPTI_ALIAS(ixptiActivityGetNextRecord, cuptiActivityGetNextRecord);
IXPTI_ALIAS(ixptiActivityGetNumDroppedRecords, cuptiActivityGetNumDroppedRecords);
IXPTI_ALIAS(ixptiActivityPopExternalCorrelationId, cuptiActivityPopExternalCorrelationId);
IXPTI_ALIAS(ixptiActivityPushExternalCorrelationId, cuptiActivityPushExternalCorrelationId);
IXPTI_ALIAS(ixptiActivityRegisterCallbacks, cuptiActivityRegisterCallbacks);
IXPTI_ALIAS(ixptiActivitySetAttribute, cuptiActivitySetAttribute);
IXPTI_ALIAS(ixptiEnableCallback, cuptiEnableCallback);
IXPTI_ALIAS(ixptiEnableDomain, cuptiEnableDomain);
IXPTI_ALIAS(ixptiFinalize, cuptiFinalize);
IXPTI_ALIAS(ixptiGetResultString, cuptiGetResultString);
IXPTI_ALIAS(ixptiGetVersion, cuptiGetVersion);
IXPTI_ALIAS(ixptiSubscribe, cuptiSubscribe);
IXPTI_ALIAS(ixptiUnsubscribe, cuptiUnsubscribe);
