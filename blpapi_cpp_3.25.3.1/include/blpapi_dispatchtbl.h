/* Copyright 2012. Bloomberg Finance L.P.
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to
 * deal in the Software without restriction, including without limitation the
 * rights to use, copy, modify, merge, publish, distribute, sublicense, and/or
 * sell copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:  The above
 * copyright notice and this permission notice shall be included in all copies
 * or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
 * FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS
 * IN THE SOFTWARE.
 */

/** \file blpapi_dispatchtbl.h */
/** \defgroup blpapi_dispatchtbl Component blpapi_dispatchtbl
\brief Work with dispatch table
\file blpapi_dispatchtbl.h
\brief Work with dispatch table
*/

#ifndef INCLUDED_BLPAPI_DISPATCHTBL
#define INCLUDED_BLPAPI_DISPATCHTBL

/** \addtogroup blpapi
 * @{
 */
/** \addtogroup blpapi_dispatchtbl
 * @{
 * <A NAME="purpose"></A>
 * <A NAME="1"> \par Purpose: </A>
 * Work with dispatch table
 * \par
 * \par
 * <A NAME="description"></A>
 * <A NAME="2"> \par Description: </A>
 *  This provides dispatch table for extended exported functions.
 * These functions are called through dispatch table based on preprocessors. It
 * avoids direct call of these functions by applications. So when blpapi DLL
 * is rolled back to the previous versions, blpapi DLL still can be loaded.
 */
/** @} */
/** @} */

#include <blpapi_correlationid.h>
#include <blpapi_defs.h>
#include <blpapi_streamproxy.h>
#include <blpapi_types.h>
#include <blpapi_versionmacros.h>

#ifdef __cplusplus

#ifndef BLPAPI_MIN_VERSION
#define BLPAPI_MIN_VERSION BLPAPI_SDK_VERSION
#endif

#ifndef BLPAPI_TARGET_VERSION
#define BLPAPI_TARGET_VERSION BLPAPI_SDK_VERSION
#endif

#if BLPAPI_MIN_VERSION > BLPAPI_TARGET_VERSION
#error "Min required version cannot be greater than target version"
#endif

extern "C" {

// Forward declarations
struct blpapi_Topic;
typedef struct blpapi_Topic blpapi_Topic_t;

struct blpapi_Message;
typedef struct blpapi_Message blpapi_Message_t;

struct blpapi_Request;
typedef struct blpapi_Request blpapi_Request_t;

struct blpapi_HighPrecisionDatetime_tag;
typedef struct blpapi_HighPrecisionDatetime_tag blpapi_HighPrecisionDatetime_t;

struct blpapi_TimePoint;
typedef struct blpapi_TimePoint blpapi_TimePoint_t;

struct blpapi_SubscriptionList;
typedef struct blpapi_SubscriptionList blpapi_SubscriptionList_t;

struct blpapi_ServiceRegistrationOptions;
typedef struct blpapi_ServiceRegistrationOptions
        blpapi_ServiceRegistrationOptions_t;

struct blpapi_RequestTemplate;
typedef struct blpapi_RequestTemplate blpapi_RequestTemplate_t;

typedef void (*blpapi_SubscriptionPreprocessErrorHandler_t)(
        const blpapi_CorrelationId_t *correlationId,
        const char *subscriptionString,
        int errorCode,
        const char *errorDescription,
        void *userData);

// End Forward declarations

// Function dispatch table declaration
typedef struct blpapi_FunctionEntries {
    int (*blpapi_EventFormatter_appendMessageSeq)(
            blpapi_EventFormatter_t *formatter,
            char const *typeString,
            blpapi_Name_t *typeName,
            const blpapi_Topic_t *topic,
            unsigned int sequenceNumber,
            unsigned int);
    int (*blpapi_EventFormatter_appendRecapMessageSeq)(
            blpapi_EventFormatter_t *formatter,
            const blpapi_Topic_t *topic,
            const blpapi_CorrelationId_t *cid,
            unsigned int sequenceNumber,
            unsigned int);
    int (*blpapi_Message_addRef)(const blpapi_Message_t *message);
    int (*blpapi_Message_release)(const blpapi_Message_t *message);
    void (*blpapi_SessionOptions_setMaxEventQueueSize)(
            blpapi_SessionOptions_t *parameters, size_t maxEventQueueSize);
    int (*blpapi_SessionOptions_setSlowConsumerWarningHiWaterMark)(
            blpapi_SessionOptions_t *parameters, float hiWaterMark);
    int (*blpapi_SessionOptions_setSlowConsumerWarningLoWaterMark)(
            blpapi_SessionOptions_t *parameters, float loWaterMark);
    void (*blpapi_Request_setPreferredRoute)(
            blpapi_Request_t *request, blpapi_CorrelationId_t *correlationId);
    int (*blpapi_Message_fragmentType)(const blpapi_Message_t *message);
    size_t (*blpapi_SessionOptions_maxEventQueueSize)(
            blpapi_SessionOptions_t *parameters);
    float (*blpapi_SessionOptions_slowConsumerWarningHiWaterMark)(
            blpapi_SessionOptions_t *parameters);
    float (*blpapi_SessionOptions_slowConsumerWarningLoWaterMark)(
            blpapi_SessionOptions_t *parameters);
    int (*blpapi_SessionOptions_setDefaultKeepAliveInactivityTime)(
            blpapi_SessionOptions_t *parameters, int inactivityTime);
    int (*blpapi_SessionOptions_setDefaultKeepAliveResponseTimeout)(
            blpapi_SessionOptions_t *parameters, int responseTimeout);
    int (*blpapi_SessionOptions_defaultKeepAliveInactivityTime)(
            blpapi_SessionOptions_t *parameters);
    int (*blpapi_SessionOptions_defaultKeepAliveResponseTimeout)(
            blpapi_SessionOptions_t *parameters);
    int (*blpapi_HighPrecisionDatetime_compare)(
            const blpapi_HighPrecisionDatetime_t *,
            const blpapi_HighPrecisionDatetime_t *);
    int (*blpapi_HighPrecisionDatetime_print)(
            const blpapi_HighPrecisionDatetime_t *,
            blpapi_StreamWriter_t,
            void *,
            int,
            int);
    int (*blpapi_Element_getValueAsHighPrecisionDatetime)(
            const blpapi_Element_t *,
            blpapi_HighPrecisionDatetime_t *,
            size_t);
    int (*blpapi_Element_setValueHighPrecisionDatetime)(blpapi_Element_t *,
            const blpapi_HighPrecisionDatetime_t *,
            size_t);
    int (*blpapi_Element_setElementHighPrecisionDatetime)(blpapi_Element_t *,
            const char *,
            const blpapi_Name_t *,
            const blpapi_HighPrecisionDatetime_t *);
    int (*blpapi_Session_resubscribeWithId)(blpapi_Session_t *,
            const blpapi_SubscriptionList_t *,
            int,
            const char *,
            int);
    int (*blpapi_EventFormatter_setValueNull)(
            blpapi_EventFormatter_t *, const char *, const blpapi_Name_t *);
    int (*blpapi_DiagnosticsUtil_memoryInfo)(char *, size_t);
    int (*blpapi_SessionOptions_setKeepAliveEnabled)(
            blpapi_SessionOptions_t *, int);
    int (*blpapi_SessionOptions_keepAliveEnabled)(blpapi_SessionOptions_t *);
    int (*blpapi_SubscriptionList_addResolved)(blpapi_SubscriptionList_t *,
            const char *,
            const blpapi_CorrelationId_t *);
    int (*blpapi_SubscriptionList_isResolvedAt)(
            blpapi_SubscriptionList_t *, int *, size_t);
    int (*blpapi_ProviderSession_deregisterService)(
            blpapi_ProviderSession_t *session, const char *serviceName);
    void (*blpapi_ServiceRegistrationOptions_setPartsToRegister)(
            blpapi_ServiceRegistrationOptions_t *session, int parts);
    int (*blpapi_ServiceRegistrationOptions_getPartsToRegister)(
            blpapi_ServiceRegistrationOptions_t *session);
    int (*blpapi_ProviderSession_deleteTopics)(
            blpapi_ProviderSession_t *session,
            const blpapi_Topic_t **topics,
            size_t numTopics);
    int (*blpapi_ProviderSession_activateSubServiceCodeRange)(
            blpapi_ProviderSession_t *session,
            const char *serviceName,
            int begin,
            int end,
            int priority);
    int (*blpapi_ProviderSession_deactivateSubServiceCodeRange)(
            blpapi_ProviderSession_t *session,
            const char *serviceName,
            int begin,
            int end);
    int (*blpapi_ServiceRegistrationOptions_addActiveSubServiceCodeRange)(
            blpapi_ServiceRegistrationOptions_t *parameters,
            int start,
            int end,
            int priority);
    void (*blpapi_ServiceRegistrationOptions_removeAllActiveSubServiceCodeRanges)(
            blpapi_ServiceRegistrationOptions_t *parameters);
    void (*blpapi_Logging_logTestMessage)(blpapi_Logging_Severity_t severity);
    const char *(*blpapi_getVersionIdentifier)();
    int (*blpapi_Message_timeReceived)(
            const blpapi_Message_t *message, blpapi_TimePoint_t *timeReceived);
    int (*blpapi_SessionOptions_recordSubscriptionDataReceiveTimes)(
            blpapi_SessionOptions_t *parameters);
    void (*blpapi_SessionOptions_setRecordSubscriptionDataReceiveTimes)(
            blpapi_SessionOptions_t *parameters, int shouldRecord);
    long long (*blpapi_TimePointUtil_nanosecondsBetween)(
            const blpapi_TimePoint_t *start, const blpapi_TimePoint_t *end);
    int (*blpapi_HighResolutionClock_now)(blpapi_TimePoint_t *timePoint);
    int (*blpapi_HighPrecisionDatetime_fromTimePoint)(
            blpapi_HighPrecisionDatetime_t *datetime,
            const blpapi_TimePoint_t *timePoint,
            short offset);
    int (*blpapi_RequestTemplate_addRef)(
            const blpapi_RequestTemplate_t *requestTemplate);
    int (*blpapi_RequestTemplate_release)(
            const blpapi_RequestTemplate_t *requestTemplate);
    int (*blpapi_Session_sendRequestTemplate)(blpapi_Session_t *session,
            const blpapi_RequestTemplate_t *requestTemplate,
            blpapi_CorrelationId_t *correlationId);
    int (*blpapi_Session_createSnapshotRequestTemplate)(
            blpapi_RequestTemplate_t **requestTemplate,
            blpapi_Session_t *session,
            const char *subscriptionString,
            const blpapi_Identity_t *identity,
            blpapi_CorrelationId_t *correlationId);
    int (*blpapi_Message_print)(const blpapi_Message_t *message,
            blpapi_StreamWriter_t streamWriter,
            void *stream,
            int indentLevel,
            int spacesPerLevel);
    int (*blpapi_Message_recapType)(const blpapi_Message_t *message);
    int (*blpapi_SessionOptions_setServiceCheckTimeout)(
            blpapi_SessionOptions_t *parameters, int timeoutMsecs);
    int (*blpapi_SessionOptions_setServiceDownloadTimeout)(
            blpapi_SessionOptions_t *parameters, int timeoutMsecs);
    int (*blpapi_SessionOptions_serviceCheckTimeout)(
            blpapi_SessionOptions_t *parameters);
    int (*blpapi_SessionOptions_serviceDownloadTimeout)(
            blpapi_SessionOptions_t *parameters);

    // 3.10.5
    int (*blpapi_ProviderSession_terminateSubscriptionsOnTopics)(
            blpapi_ProviderSession_t *session,
            const blpapi_Topic_t **topics,
            size_t numTopics,
            const char *message);

    // 3.10.8
    int (*blpapi_EventFormatter_appendFragmentedRecapMessage)(
            blpapi_EventFormatter_t *formatter,
            const char *typeString,
            blpapi_Name_t *typeName,
            const blpapi_Topic_t *topic,
            const blpapi_CorrelationId_t *cid,
            int fragmentType);
    int (*blpapi_EventFormatter_appendFragmentedRecapMessageSeq)(
            blpapi_EventFormatter_t *formatter,
            const char *typeString,
            blpapi_Name_t *typeName,
            const blpapi_Topic_t *topic,
            int fragmentType,
            unsigned int sequenceNumber);

    // 3.11.0
    void (*blpapi_SessionOptions_setTlsOptions)(
            blpapi_SessionOptions_t *parameters,
            const blpapi_TlsOptions_t *tlsOptions);
    blpapi_TlsOptions_t *(*blpapi_TlsOptions_createFromFiles)(
            const char *clientCredentialsFileName,
            const char *clientCredentialsPassword,
            const char *trustedCertificatesFileName);
    blpapi_TlsOptions_t *(*blpapi_TlsOptions_createFromBlobs)(
            const char *clientCredentialsRawData,
            int clientCredentialsRawDataLength,
            const char *clientCredentialsPassword,
            const char *trustedCertificatesRawData,
            int trustedCertificatesRawDataLength);
    void (*blpapi_TlsOptions_setTlsHandshakeTimeoutMs)(
            blpapi_TlsOptions_t *paramaters, int tlsHandshakeTimeoutMs);
    void (*blpapi_TlsOptions_setCrlFetchTimeoutMs)(
            blpapi_TlsOptions_t *paramaters, int crlFetchTimeoutMs);

    blpapi_TlsOptions_t *(*blpapi_TlsOptions_create)(void);
    blpapi_TlsOptions_t *(*blpapi_TlsOptions_duplicate)(
            const blpapi_TlsOptions_t *parameters);
    void (*blpapi_TlsOptions_copy)(
            blpapi_TlsOptions_t *lhs, const blpapi_TlsOptions_t *rhs);
    void (*blpapi_TlsOptions_destroy)(blpapi_TlsOptions_t *parameters);

    // 3.11.2
    int (*blpapi_AbstractSession_generateManualToken)(
            blpapi_AbstractSession_t *session,
            blpapi_CorrelationId_t *correlationId,
            const char *user,
            const char *manualIp,
            blpapi_EventQueue_t *eventQueue);

    // 3.11.4
    int (*blpapi_EventFormatter_appendValueHighPrecisionDatetime)(
            blpapi_EventFormatter_t *formatter,
            const blpapi_HighPrecisionDatetime_t *value);
    int (*blpapi_EventFormatter_setValueHighPrecisionDatetime)(
            blpapi_EventFormatter_t *formatter,
            const char *typeString,
            const blpapi_Name_t *typeName,
            const blpapi_HighPrecisionDatetime_t *value);

    // 3.12.0
    int (*blpapi_SessionOptions_print)(blpapi_SessionOptions_t *parameters,
            blpapi_StreamWriter_t streamWriter,
            void *stream,
            int indentLevel,
            int spacesPerLevel);
    int (*blpapi_SessionOptions_flushPublishedEventsTimeout)(
            blpapi_SessionOptions_t *parameters);
    int (*blpapi_SessionOptions_setFlushPublishedEventsTimeout)(
            blpapi_SessionOptions_t *parameters, int timeoutMsecs);
    int (*blpapi_ProviderSession_flushPublishedEvents)(
            blpapi_ProviderSession_t *session,
            int *allFlushed,
            int timeoutMsecs);

    // 3.13.0
    int (*blpapi_ZfpUtil_getOptionsForLeasedLines)(
            blpapi_SessionOptions_t *sessionOptions,
            const blpapi_TlsOptions_t *tlsOptions,
            int remote);

    // 3.14.0
    int (*blpapi_SessionOptions_setBandwidthSaveModeDisabled)(
            blpapi_SessionOptions_t *parameters, int disableBandwidthSaveMode);

    int (*blpapi_SessionOptions_bandwidthSaveModeDisabled)(
            blpapi_SessionOptions_t *parameters);

    // 3.14.1
    int (*blpapi_TestUtil_deserializeService)(const char *schema,
            size_t schemaLength,
            blpapi_Service_t **service);

    int (*blpapi_TestUtil_serializeService)(blpapi_StreamWriter_t streamWriter,
            void *userStream,
            const blpapi_Service_t *service);

    int (*blpapi_TestUtil_createTopic)(blpapi_Topic_t **topic,
            const blpapi_Service_t *service,
            int isActive);

    int (*blpapi_TestUtil_getAdminMessageDefinition)(
            blpapi_SchemaElementDefinition_t **definition,
            blpapi_Name_t *messageName);

    int (*blpapi_TestUtil_createEvent)(blpapi_Event_t **event, int eventType);

    int (*blpapi_TestUtil_appendMessage)(blpapi_MessageFormatter_t **formatter,
            blpapi_Event_t *event,
            const blpapi_SchemaElementDefinition_t *messageType,
            const blpapi_MessageProperties_t *properties);

    int (*blpapi_MessageProperties_create)(
            blpapi_MessageProperties_t **messageProperties);

    void (*blpapi_MessageProperties_destroy)(
            blpapi_MessageProperties_t *messageProperties);

    int (*blpapi_MessageProperties_copy)(blpapi_MessageProperties_t **dest,
            const blpapi_MessageProperties_t *src);

    int (*blpapi_MessageProperties_assign)(blpapi_MessageProperties_t *lhs,
            const blpapi_MessageProperties_t *rhs);

    int (*blpapi_MessageProperties_setCorrelationIds)(
            blpapi_MessageProperties_t *messageProperties,
            const blpapi_CorrelationId_t *correlationIds,
            size_t numCorrelationIds);

    int (*blpapi_MessageProperties_setRecapType)(
            blpapi_MessageProperties_t *messageProperties,
            int recap,
            int fragment);

    int (*blpapi_MessageProperties_setTimeReceived)(
            blpapi_MessageProperties_t *messageProperties,
            const blpapi_HighPrecisionDatetime_t *timestamp);

    int (*blpapi_MessageProperties_setService)(
            blpapi_MessageProperties_t *messageProperties,
            const blpapi_Service_t *service);

    int (*blpapi_MessageFormatter_setValueBool)(
            blpapi_MessageFormatter_t *formatter,
            const blpapi_Name_t *typeName,
            blpapi_Bool_t value);

    int (*blpapi_MessageFormatter_setValueChar)(
            blpapi_MessageFormatter_t *formatter,
            const blpapi_Name_t *typeName,
            char value);

    int (*blpapi_MessageFormatter_setValueInt32)(
            blpapi_MessageFormatter_t *formatter,
            const blpapi_Name_t *typeName,
            blpapi_Int32_t value);

    int (*blpapi_MessageFormatter_setValueInt64)(
            blpapi_MessageFormatter_t *formatter,
            const blpapi_Name_t *typeName,
            blpapi_Int64_t value);

    int (*blpapi_MessageFormatter_setValueFloat32)(
            blpapi_MessageFormatter_t *formatter,
            const blpapi_Name_t *typeName,
            blpapi_Float32_t value);

    int (*blpapi_MessageFormatter_setValueFloat64)(
            blpapi_MessageFormatter_t *formatter,
            const blpapi_Name_t *typeName,
            blpapi_Float64_t value);

    int (*blpapi_MessageFormatter_setValueDatetime)(
            blpapi_MessageFormatter_t *formatter,
            const blpapi_Name_t *typeName,
            const blpapi_Datetime_t *value);

    int (*blpapi_MessageFormatter_setValueHighPrecisionDatetime)(
            blpapi_MessageFormatter_t *formatter,
            const blpapi_Name_t *typeName,
            const blpapi_HighPrecisionDatetime_t *value);

    int (*blpapi_MessageFormatter_setValueString)(
            blpapi_MessageFormatter_t *formatter,
            const blpapi_Name_t *typeName,
            const char *value);

    int (*blpapi_MessageFormatter_setValueFromName)(
            blpapi_MessageFormatter_t *formatter,
            const blpapi_Name_t *typeName,
            const blpapi_Name_t *value);

    int (*blpapi_MessageFormatter_setValueNull)(
            blpapi_MessageFormatter_t *formatter,
            const blpapi_Name_t *typeName);

    int (*blpapi_MessageFormatter_pushElement)(
            blpapi_MessageFormatter_t *formatter,
            const blpapi_Name_t *typeName);

    int (*blpapi_MessageFormatter_popElement)(
            blpapi_MessageFormatter_t *formatter);

    int (*blpapi_MessageFormatter_appendValueBool)(
            blpapi_MessageFormatter_t *formatter, blpapi_Bool_t value);

    int (*blpapi_MessageFormatter_appendValueChar)(
            blpapi_MessageFormatter_t *formatter, char value);

    int (*blpapi_MessageFormatter_appendValueInt32)(
            blpapi_MessageFormatter_t *formatter, blpapi_Int32_t value);

    int (*blpapi_MessageFormatter_appendValueInt64)(
            blpapi_MessageFormatter_t *formatter, blpapi_Int64_t value);

    int (*blpapi_MessageFormatter_appendValueFloat32)(
            blpapi_MessageFormatter_t *formatter, blpapi_Float32_t value);

    int (*blpapi_MessageFormatter_appendValueFloat64)(
            blpapi_MessageFormatter_t *formatter, blpapi_Float64_t value);

    int (*blpapi_MessageFormatter_appendValueDatetime)(
            blpapi_MessageFormatter_t *formatter,
            const blpapi_Datetime_t *value);

    int (*blpapi_MessageFormatter_appendValueHighPrecisionDatetime)(
            blpapi_MessageFormatter_t *formatter,
            const blpapi_HighPrecisionDatetime_t *value);

    int (*blpapi_MessageFormatter_appendValueString)(
            blpapi_MessageFormatter_t *formatter, const char *value);

    int (*blpapi_MessageFormatter_appendValueFromName)(
            blpapi_MessageFormatter_t *formatter, const blpapi_Name_t *value);

    int (*blpapi_MessageFormatter_appendElement)(
            blpapi_MessageFormatter_t *formatter);

    int (*blpapi_MessageFormatter_FormatMessageJson)(
            blpapi_MessageFormatter_t *formatter, const char *message);

    int (*blpapi_MessageFormatter_FormatMessageXml)(
            blpapi_MessageFormatter_t *formatter, const char *message);

    int (*blpapi_MessageFormatter_copy)(blpapi_MessageFormatter_t **formatter,
            const blpapi_MessageFormatter_t *original);

    int (*blpapi_MessageFormatter_assign)(blpapi_MessageFormatter_t **lhs,
            const blpapi_MessageFormatter_t *rhs);

    int (*blpapi_MessageFormatter_destroy)(
            blpapi_MessageFormatter_t *formatter);

    int (*blpapi_Operation_responseDefinitionFromName)(
            blpapi_Operation_t *operation,
            blpapi_SchemaElementDefinition_t **responseDefinition,
            const blpapi_Name_t *name);

    // 3.15.0
    int (*blpapi_SessionOptions_setSessionIdentityOptions)(
            blpapi_SessionOptions_t *parameters,
            const blpapi_AuthOptions_t *authOptions,
            blpapi_CorrelationId_t *cid);

    int (*blpapi_AbstractSession_generateAuthorizedIdentityAsync)(
            blpapi_AbstractSession_t *session,
            const blpapi_AuthOptions_t *authOptions,
            blpapi_CorrelationId_t *cid);

    int (*blpapi_AbstractSession_getAuthorizedIdentity)(
            blpapi_AbstractSession_t *session,
            const blpapi_CorrelationId_t *cid,
            blpapi_Identity_t **identity);

    int (*blpapi_AuthOptions_create_default)(blpapi_AuthOptions_t **options);

    int (*blpapi_AuthOptions_create_forUserMode)(
            blpapi_AuthOptions_t **options, const blpapi_AuthUser_t *user);

    int (*blpapi_AuthOptions_create_forAppMode)(blpapi_AuthOptions_t **options,
            const blpapi_AuthApplication_t *app);

    int (*blpapi_AuthOptions_create_forUserAndAppMode)(
            blpapi_AuthOptions_t **options,
            const blpapi_AuthUser_t *user,
            const blpapi_AuthApplication_t *app);

    int (*blpapi_AuthOptions_create_forToken)(
            blpapi_AuthOptions_t **options, const blpapi_AuthToken_t *token);

    int (*blpapi_AuthOptions_duplicate)(
            blpapi_AuthOptions_t **options, const blpapi_AuthOptions_t *dup);

    int (*blpapi_AuthOptions_copy)(
            blpapi_AuthOptions_t *lhs, const blpapi_AuthOptions_t *rhs);

    void (*blpapi_AuthOptions_destroy)(blpapi_AuthOptions_t *options);

    int (*blpapi_AuthUser_createWithLogonName)(blpapi_AuthUser_t **user);

    int (*blpapi_AuthUser_createWithActiveDirectoryProperty)(
            blpapi_AuthUser_t **user, const char *propertyName);

    int (*blpapi_AuthUser_createWithManualOptions)(blpapi_AuthUser_t **user,
            const char *userId,
            const char *ipAddress);

    int (*blpapi_AuthUser_duplicate)(
            blpapi_AuthUser_t **user, const blpapi_AuthUser_t *dup);

    int (*blpapi_AuthUser_copy)(
            blpapi_AuthUser_t *lhs, const blpapi_AuthUser_t *rhs);

    void (*blpapi_AuthUser_destroy)(blpapi_AuthUser_t *user);

    int (*blpapi_AuthApplication_create)(
            blpapi_AuthApplication_t **app, const char *appName);

    int (*blpapi_AuthApplication_duplicate)(blpapi_AuthApplication_t **app,
            const blpapi_AuthApplication_t *dup);

    int (*blpapi_AuthApplication_copy)(blpapi_AuthApplication_t *lhs,
            const blpapi_AuthApplication_t *rhs);

    void (*blpapi_AuthApplication_destroy)(blpapi_AuthApplication_t *app);

    int (*blpapi_AuthToken_create)(
            blpapi_AuthToken_t **token, const char *tokenStr);

    int (*blpapi_AuthToken_duplicate)(
            blpapi_AuthToken_t **token, const blpapi_AuthToken_t *dup);

    int (*blpapi_AuthToken_copy)(
            blpapi_AuthToken_t *lhs, const blpapi_AuthToken_t *rhs);

    void (*blpapi_AuthToken_destroy)(blpapi_AuthToken_t *token);

    // 3.16.0
    int (*blpapi_Message_getRequestId)(
            const blpapi_Message_t *message, const char **requestId);
    int (*blpapi_Request_getRequestId)(
            const blpapi_Request_t *request, const char **requestId);
    int (*blpapi_MessageProperties_setRequestId)(
            blpapi_MessageProperties_t *messageProperties,
            const char *requestId);

    // 3.18.0
    int (*blpapi_Session_subscribeEx)(blpapi_Session_t *session,
            const blpapi_SubscriptionList_t *subscriptionList,
            const blpapi_Identity_t *handle,
            const char *requestLabel,
            int requestLabelLen,
            blpapi_SubscriptionPreprocessErrorHandler_t errorHandler,
            void *userData);
    int (*blpapi_Session_resubscribeEx)(blpapi_Session_t *session,
            const blpapi_SubscriptionList_t *resubscriptionList,
            const char *requestLabel,
            int requestLabelLen,
            blpapi_SubscriptionPreprocessErrorHandler_t errorHandler,
            void *userData);
    int (*blpapi_Session_resubscribeWithIdEx)(blpapi_Session_t *session,
            const blpapi_SubscriptionList_t *resubscriptionList,
            int resubscriptionId,
            const char *requestLabel,
            int requestLabelLen,
            blpapi_SubscriptionPreprocessErrorHandler_t errorHandler,
            void *userData);

    // 3.18.5
    int (*blpapi_SessionOptions_applicationIdentityKey)(
            const char **applicationIdentityKey,
            size_t *size,
            blpapi_SessionOptions_t *parameters);
    int (*blpapi_SessionOptions_setApplicationIdentityKey)(
            blpapi_SessionOptions_t *parameters,
            const char *applicationIdentityKey,
            unsigned size);

    // 3.19
    int (*blpapi_Element_getValueAsBytes)(const blpapi_Element_t *element,
            const char **buffer,
            size_t *length,
            size_t index);

    int (*blpapi_Element_setValueBytes)(blpapi_Element_t *element,
            const char *value,
            size_t length,
            size_t index);

    int (*blpapi_Element_setElementBytes)(blpapi_Element_t *element,
            const char *nameString,
            const blpapi_Name_t *name,
            const char *value,
            size_t length);

    int (*blpapi_EventFormatter_setValueBytes)(
            blpapi_EventFormatter_t *formatter,
            const char *typeString,
            const blpapi_Name_t *typeName,
            const char *value,
            size_t length);

    int (*blpapi_MessageFormatter_setValueBytes)(
            blpapi_MessageFormatter_t *formatter,
            const blpapi_Name_t *typeName,
            const char *value,
            size_t length);

    // 3.20.0
    blpapi_Socks5Config_t *(*blpapi_Socks5Config_create)(
            const char *hostname, size_t hostname_size, unsigned short port);

    int (*blpapi_Socks5Config_copy)(blpapi_Socks5Config_t **socks5Config,
            const blpapi_Socks5Config_t *srcSocks5Config);

    void (*blpapi_Socks5Config_destroy)(blpapi_Socks5Config_t *socks5Config);

    int (*blpapi_Socks5Config_print)(blpapi_Socks5Config_t *socks5Config,
            blpapi_StreamWriter_t streamWriter,
            void *userStream,
            int indentLevel,
            int spacesPerLevel);

    int (*blpapi_SessionOptions_getServerAddressWithProxy)(
            blpapi_SessionOptions_t *parameters,
            const char **serverHost,
            unsigned short *serverPort,
            const char **socks5Host,
            unsigned short *sock5Port,
            size_t index);

    int (*blpapi_SessionOptions_setServerAddressWithProxy)(
            blpapi_SessionOptions_t *parameters,
            const char *serverHost,
            unsigned short serverPort,
            const blpapi_Socks5Config_t *socks5Config,
            size_t index);

    // 3.22.0
    int (*blpapi_SessionOptions_sessionName)(const char **sessionName,
            size_t *size,
            blpapi_SessionOptions_t *parameters);

    int (*blpapi_SessionOptions_setSessionName)(
            blpapi_SessionOptions_t *parameters,
            const char *sessionName,
            size_t size);

    int (*blpapi_AbstractSession_sessionName)(
            blpapi_AbstractSession_t *session,
            const char **sessionName,
            size_t *size);

    int (*blpapi_MessageIterator_addRef)(
            const blpapi_MessageIterator_t *iterator);

    // 3.24.0
    int (*blpapi_EventFormatter_getElementDefinition)(
            blpapi_EventFormatter_t *formatter,
            blpapi_SchemaElementDefinition_t **definition);

    int (*blpapi_Logging_configureLogging)(int level,
            const char *fileName,
            int rolloverFileCount,
            int rolloverFileLimit);

    // 3.24.7
    int (*blpapi_Session_addRef)(const blpapi_Session_t *session);
    int (*blpapi_Session_release)(const blpapi_Session_t *session);
    int (*blpapi_ProviderSession_addRef)(
            const blpapi_ProviderSession_t *session);
    int (*blpapi_ProviderSession_release)(
            const blpapi_ProviderSession_t *session);

    // 3.24.12
    int (*blpapi_UserAgentInfo_setUserTaskName)(const char *userTaskName);
    int (*blpapi_UserAgentInfo_setNativeSdkLanguageAndVersion)(
            const char *language, const char *version);

} blpapi_FunctionEntries_t;

BLPAPI_EXPORT extern size_t g_blpapiFunctionTableSize;
BLPAPI_EXPORT extern blpapi_FunctionEntries_t g_blpapiFunctionEntries;

} // extern "C"

#endif // __cplusplus

#endif // INCLUDED_BLPAPI_DISPATCHTBL
