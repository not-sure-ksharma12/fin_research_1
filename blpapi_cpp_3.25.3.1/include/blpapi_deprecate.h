// blpapi_deprecate.h                           -*-C++-*-
#ifndef INCLUDED_BLPAPI_DEPRECATE
#define INCLUDED_BLPAPI_DEPRECATE

//@PURPOSE: Provide (generated) deprecation facilities.
//
//@MACROS:
//  BLPAPI_DEPRECATE_MESSAGE_TOPIC_NAME
//  BLPAPI_DEPRECATE_MESSAGE_TOPIC_NAME_MSG(MSG)
//  BLPAPI_DEPRECATE_ABSTRACT_SESSION_CREATE_USER_HANDLE
//  BLPAPI_DEPRECATE_ABSTRACT_SESSION_CREATE_USER_HANDLE_MSG(MSG)
//  BLPAPI_DEPRECATE_STRING_NAME
//  BLPAPI_DEPRECATE_STRING_NAME_MSG(MSG)
//  BLPAPI_DEPRECATE_PRERESOLVED_TOPICS
//  BLPAPI_DEPRECATE_PRERESOLVED_TOPICS_MSG(MSG)
//
//@DESCRIPTION: This **generated** header provides a collection of public
// macros allowing code in this package to be marked deprecated, as
// well as defining a set of macros the user may supply (typically as compiler
// options) to configure the behavior of the deprecation annotations.  These
// deprecation annotations may, depending on the configuration, instantiate as
// C++ ''[[deprecated]]'' annotations for which the compiler will emit a
// warning.  This component allows for the configuration of the behavior of the
// instantiation of the macros to allow users to customize the set of warnings
// that are received as part of a build.  So, for example, a user working on
// local development may customize their build to warn them whenever they use
// any deprecated feature, configure their CI systems to warn only for more
// mature deprecations (i.e., older deprecations), and finally configure their
// production build process to  produce no warnings at all for deprecations.
//
// This file is generated based on an associated configuration file that
// provides the set of features in this package that are deprecated, as well
// as additional metadata about those deprecates (such as the stage and
// severity of the deprecation).
//
/// Deprecated Feature Reference
///----------------------------
// Below is a list of the deprecated features for which this component
// provides deprecation annotations, as well as the severity and state of
// those deprecated features.  This information is generated from a
// deprecation configuration file associated with this package.  Additional
// information is available here
// https://bbgithub.dev.bloomberg.com/cpp-guild/deprecations/
//
//..
//  FEATURE: deprecation-stage, severity
//  -----------------------------------
//  MESSAGE_TOPIC_NAME: default_warning, standard
//  ABSTRACT_SESSION_CREATE_USER_HANDLE: default_warning, standard
//  STRING_NAME: default_warning, standard
//  PRERESOLVED_TOPICS: default_warning, standard
//..
//
/// Macro Reference
///---------------
// For each deprecated feature ("[FEATURE]") in this package two macros are
// provided (see {'Deprecated Feature Reference'} for a list of features):
//
//: 'BLPAPI_DEPRECATE_[FEATURE]':
//:    This macro is used to annotate code to indicate that a name or entity
//:    has been deprecated.  This macro can be used as if it were the C++14
//:    standard attribute '[[deprecated]]', and in appropriate build
//:    configurations will instantiate as a C++ '[[deprecated]]' annotation.
//:    The compiler warning message generated from this annotation will
//:    contain descriptive text based on the metadata associated with the
//:    deprecated feature.
//:
//: 'BLPAPI_DEPRECATE_[FEATURE]_MSG(MSG)':
//:    This macro is used to annotate code to indicate that a name or entity
//:    has been deprecated and to convey the specified, instance specific,
//:    deprecation 'MSG'.  This macro is similar to
//:    'BLPAPI_DEPRECATE_[FEATURE]' but appends the supplied 'MSG' to the
//:    deprecation warning, which can be used to indicate information specific
//:    to the name or entity being annotated (as more than one entity may be
//:    marked deprecated as part of deprecating a feature).  This macro
//:    can be used as-if it were the C++14 standard attribute '[[deprecated]]',
//:    and in appropriate build configurations will instantiate as a C++
//:    '[[deprecated]]' annotation.  The compiler warning message generated
//:    from this annotation will contain descriptive text based on the
//:    metadata associated with the deprecated feature, as well as the
//:    supplied 'MSG'.
//
// By default, the deprecation stage for a feature determines whether a
// deprecation annotation macro will be instantiated as a C++ '[[deprecation]]'
// annotation.  A deprecation in the 'default_no_warning' stage will *not*
// generate a C++ deprecation annotation *by* *default* (the macro is simply
// ignored), a deprecation in the 'default_warning' or 'removal' stage will
// generate a C++ '[[deprecation]]' annotation (which will result in
// compile-time warnings if the entity is used).  Users may configure the
// behavior of these annotations at compile time using macros described in the
// {'Configuration Reference'} section below.
//
/// Macros Active By Default
///  - - - - - - - - - - - -
//: o 'BLPAPI_DEPRECATE_MESSAGE_TOPIC_NAME'
//: o 'BLPAPI_DEPRECATE_MESSAGE_TOPIC_NAME_MSG(MSG)'
//: o 'BLPAPI_DEPRECATE_ABSTRACT_SESSION_CREATE_USER_HANDLE'
//: o 'BLPAPI_DEPRECATE_ABSTRACT_SESSION_CREATE_USER_HANDLE_MSG(MSG)'
//: o 'BLPAPI_DEPRECATE_STRING_NAME'
//: o 'BLPAPI_DEPRECATE_STRING_NAME_MSG(MSG)'
//: o 'BLPAPI_DEPRECATE_PRERESOLVED_TOPICS'
//: o 'BLPAPI_DEPRECATE_PRERESOLVED_TOPICS_MSG(MSG)'
//
/// Configuration Reference
///-----------------------
// There are a set of macros, not defined by this component, that users may
// supply (e.g., to their build system) to configure the behavior of the
// deprecation annotation macros provided by this component.
//
// The available configuration macros are described below (note that
// "[FEATURE]" indicates a configuration macro exists for each deprecated
// feature managed by this component, see {'Deprecated Feature Reference'}):
//
// * 'BB_DEPRECATE_ENABLE_ALL_DEPRECATIONS_FOR_TESTING': This macro, when
//   defined, enables the instantiation of every deprecation macro as a C++
//   '[[deprecated]]' annotation.  This *MUST* *NOT* be defined when building
//   in an integrated build context, like the dpkg unstable build or ROBO.
//
// * BB_DEPRECATE_ENABLE_JSON_MESSAGE: Changes the messages reported by
//   compiler deprecation annotations to be a JSON deprecation configuration
//   text.
//
// * 'BLPAPI_DEPRECATE_ENABLE_[FEATURE]': This macro, when defined, enables
//   the instantiation of deprecations for the FEATURE.
//
// * 'BLPAPI_DEPRECATE_DISABLE_[FEATURE]': This macro, when defined, disables
//   the instantiation of deprecations for the FEATURE.
//
// * 'BLPAPI_DEPRECATE_DISABLE_ALL': This macro, when defined, disables
//   all the instantiation of deprecations configured by this header.
//
/// Configuration Precedence
///  - - - - - - - - - - - -
// Conflicts between the configuration macros above (e.g., if configuration
// macros are provided that both enable and disable a annotation) are resolved
// in order or precedence.
//
//: 1 'BB_DEPRECATE_ENABLE_ALL_DEPRECATIONS_FOR_TESTING'
//: 2 'BLPAPI_DEPRECATE_ENABLE_[FEATURE]' and
//:   'BLPAPI_DEPRECATE_DISABLE_[FEATURE]'
//: 3 'BLPAPI_DEPRECATE_DISABLE_ALL'
//
// Specifying *both* 'BLPAPI_DEPRECATE_ENABLE_[FEATURE]' and
// 'BLPAPI_DEPRECATE_DISABLE_[FEATURE]' will result in a compile time error.
//
// Notice that 'BB_DEPRECATE_ENABLE_ALL_DEPRECATIONS_FOR_TESTING' takes
// precedence because it is used by tools to gain visibility for all the
// deprecated entities in the source code.  'BLPAPI_DEPRECATE_ENABLE_[FEATURE]'
// takes precedence over 'BLPAPI_DEPRECATE_DISABLE_ALL' because it is more
// specific (overriding the more general configuration).

// ==========================================
// Common Implementation Configuration Macros
// ==========================================

#if (defined(__cplusplus) && (__cplusplus >= 201402L))                        \
        || (defined(_MSVC_LANG) && (_MSVC_LANG >= 201402L))
#define BLPAPI_DEPRECATE_SUPPORTED_PLATFORM
#endif

#ifdef BLPAPI_DEPRECATE_SUPPORTED_PLATFORM
#define BLPAPI_DEPRECATE_IMP(SEVERITY, URL, MESSAGE)                          \
    [[deprecated("severity=\"" SEVERITY "\", url=\"" URL                      \
                 "\", message=\"" MESSAGE "\"")]]
#define BLPAPI_DEPRECATE_IMP_CONFIG(CONFIG_MSG) [[deprecated(CONFIG_MSG)]]
#else
#define BLPAPI_DEPRECATE_IMP(SEVERITY, URL, MESSAGE)
#define BLPAPI_DEPRECATE_IMP_CONFIG(CONFIG_MSG)
#endif

// Define a set of standard threshold levels.
#define BLPAPI_DEPRECATE_SEVERITY_LEVEL_NOTIFICATION 1
#define BLPAPI_DEPRECATE_SEVERITY_LEVEL_STANDARD 64
#define BLPAPI_DEPRECATE_SEVERITY_LEVEL_CRITICAL 128

// Determine the threshold level for generating [[deprecated]] warnings
#ifndef BLPAPI_DEPRECATE_SEVERITY_THRESHOLD
#define BLPAPI_DEPRECATE_SEVERITY_THRESHOLD 0
#endif

// Define a set of standard deprecation stages
#define BLPAPI_DEPRECATE_STAGE_LEVEL_DEFAULT_NO_WARNING 64
#define BLPAPI_DEPRECATE_STAGE_LEVEL_DEFAULT_WARNING 128
#define BLPAPI_DEPRECATE_STAGE_LEVEL_REMOVAL 254

// Define a default deprecation stage threshold
#ifndef BLPAPI_DEPRECATE_STAGE_THRESHOLD
#define BLPAPI_DEPRECATE_STAGE_THRESHOLD                                      \
    BLPAPI_DEPRECATE_STAGE_LEVEL_DEFAULT_WARNING
#endif

// =============================
// Macros for MESSAGE_TOPIC_NAME
// =============================

#if defined(BLPAPI_DEPRECATE_MESSAGE_TOPIC_NAME__ENABLED)
#error "BLPAPI_DEPRECATE_MESSAGE_TOPIC_NAME__ENABLED is defined outside of the component."
#endif

#if defined(BLPAPI_DEPRECATE_ENABLE_MESSAGE_TOPIC_NAME)                       \
        && defined(BLPAPI_DEPRECATE_DISABLE_MESSAGE_TOPIC_NAME)
#error "Both BLPAPI_DEPRECATE_ENABLE_MESSAGE_TOPIC_NAME and BLPAPI_DEPRECATE_DISABLE_MESSAGE_TOPIC_NAME are enabled."
#endif

#if defined(BB_DEPRECATE_ENABLE_ALL_DEPRECATIONS_FOR_TESTING)
#define BLPAPI_DEPRECATE_MESSAGE_TOPIC_NAME__ENABLED 1
#elif defined(BLPAPI_DEPRECATE_ENABLE_MESSAGE_TOPIC_NAME)
#define BLPAPI_DEPRECATE_MESSAGE_TOPIC_NAME__ENABLED 1
#elif (BLPAPI_DEPRECATE_STAGE_THRESHOLD                                       \
        <= BLPAPI_DEPRECATE_STAGE_LEVEL_DEFAULT_WARNING)                      \
        && (BLPAPI_DEPRECATE_SEVERITY_THRESHOLD                               \
                <= BLPAPI_DEPRECATE_SEVERITY_LEVEL_STANDARD)                  \
        && (!defined(BLPAPI_DEPRECATE_DISABLE_MESSAGE_TOPIC_NAME))            \
        && (!defined(BLPAPI_DEPRECATE_DISABLE_ALL))
#define BLPAPI_DEPRECATE_MESSAGE_TOPIC_NAME__ENABLED 1
#endif // defined(BB_DEPRECATE_ENABLE_ALL_DEPRECATIONS_FOR_TESTING)

#define BLPAPI_DEPRECATE_MESSAGE_TOPIC_NAME_JSON                              \
    R"text({"feature": "message_topic_name", "description": "Deprecating blpapi::Message::topicName() method", "severity": "standard", "stage": "default_warning", "url": "https://bloomberg.github.io/blpapi-docs/cpp/3.24.4/classBloombergLP_1_1blpapi_1_1Message.html#a4ecdba068f788562c4cc919a735d253b", "message": "This method always returns an empty string", "library": "blpapi", "group": "blpapi"})text"

#if defined(BLPAPI_DEPRECATE_MESSAGE_TOPIC_NAME__ENABLED)
#if defined(BB_DEPRECATE_ENABLE_JSON_MESSAGE)
#define BLPAPI_DEPRECATE_MESSAGE_TOPIC_NAME                                   \
    BLPAPI_DEPRECATE_IMP_CONFIG(BLPAPI_DEPRECATE_MESSAGE_TOPIC_NAME_JSON)
#define BLPAPI_DEPRECATE_MESSAGE_TOPIC_NAME_MSG(MSG)                          \
    BLPAPI_DEPRECATE_MESSAGE_TOPIC_NAME
#else
#define BLPAPI_DEPRECATE_MESSAGE_TOPIC_NAME                                   \
    BLPAPI_DEPRECATE_IMP("standard",                                          \
            "https://bloomberg.github.io/blpapi-docs/cpp/3.24.4/"             \
            "classBloombergLP_1_1blpapi_1_1Message.html#"                     \
            "a4ecdba068f788562c4cc919a735d253b",                              \
            "This method always returns an empty string")
#define BLPAPI_DEPRECATE_MESSAGE_TOPIC_NAME_MSG(MSG)                          \
    BLPAPI_DEPRECATE_IMP("standard",                                          \
            "https://bloomberg.github.io/blpapi-docs/cpp/3.24.4/"             \
            "classBloombergLP_1_1blpapi_1_1Message.html#"                     \
            "a4ecdba068f788562c4cc919a735d253b",                              \
            "This method always returns an empty string: " MSG)
#endif // defined(BB_DEPRECATE_ENABLE_JSON_MESSAGE)
#endif // defined(BLPAPI_DEPRECATE_MESSAGE_TOPIC_NAME__ENABLED)

#if !defined(BLPAPI_DEPRECATE_MESSAGE_TOPIC_NAME)
#define BLPAPI_DEPRECATE_MESSAGE_TOPIC_NAME
#define BLPAPI_DEPRECATE_MESSAGE_TOPIC_NAME_MSG(MSG)
#endif // !defined(BLPAPI_DEPRECATE_MESSAGE_TOPIC_NAME)

// ==============================================
// Macros for ABSTRACT_SESSION_CREATE_USER_HANDLE
// ==============================================

#if defined(BLPAPI_DEPRECATE_ABSTRACT_SESSION_CREATE_USER_HANDLE__ENABLED)
#error "BLPAPI_DEPRECATE_ABSTRACT_SESSION_CREATE_USER_HANDLE__ENABLED is defined outside of the component."
#endif

#if defined(BLPAPI_DEPRECATE_ENABLE_ABSTRACT_SESSION_CREATE_USER_HANDLE)      \
        && defined(                                                           \
                BLPAPI_DEPRECATE_DISABLE_ABSTRACT_SESSION_CREATE_USER_HANDLE)
#error "Both BLPAPI_DEPRECATE_ENABLE_ABSTRACT_SESSION_CREATE_USER_HANDLE and BLPAPI_DEPRECATE_DISABLE_ABSTRACT_SESSION_CREATE_USER_HANDLE are enabled."
#endif

#if defined(BB_DEPRECATE_ENABLE_ALL_DEPRECATIONS_FOR_TESTING)
#define BLPAPI_DEPRECATE_ABSTRACT_SESSION_CREATE_USER_HANDLE__ENABLED 1
#elif defined(BLPAPI_DEPRECATE_ENABLE_ABSTRACT_SESSION_CREATE_USER_HANDLE)
#define BLPAPI_DEPRECATE_ABSTRACT_SESSION_CREATE_USER_HANDLE__ENABLED 1
#elif (BLPAPI_DEPRECATE_STAGE_THRESHOLD                                        \
        <= BLPAPI_DEPRECATE_STAGE_LEVEL_DEFAULT_WARNING)                       \
        && (BLPAPI_DEPRECATE_SEVERITY_THRESHOLD                                \
                <= BLPAPI_DEPRECATE_SEVERITY_LEVEL_STANDARD)                   \
        && (!defined(                                                          \
                BLPAPI_DEPRECATE_DISABLE_ABSTRACT_SESSION_CREATE_USER_HANDLE)) \
        && (!defined(BLPAPI_DEPRECATE_DISABLE_ALL))
#define BLPAPI_DEPRECATE_ABSTRACT_SESSION_CREATE_USER_HANDLE__ENABLED 1
#endif // defined(BB_DEPRECATE_ENABLE_ALL_DEPRECATIONS_FOR_TESTING)

#define BLPAPI_DEPRECATE_ABSTRACT_SESSION_CREATE_USER_HANDLE_JSON             \
    R"text({"feature": "abstract_session_create_user_handle", "description": "Deprecating blpapi::AbstractSession::createUserHandle() method", "severity": "standard", "stage": "default_warning", "url": "https://bloomberg.github.io/blpapi-docs/cpp/3.24.4/classBloombergLP_1_1blpapi_1_1AbstractSession.html#a5e2f9d667e3f1d0bf4891268919d3e83", "message": "Use blpapi::AbstractSession::createIdentity() instead", "library": "blpapi", "group": "blpapi"})text"

#if defined(BLPAPI_DEPRECATE_ABSTRACT_SESSION_CREATE_USER_HANDLE__ENABLED)
#if defined(BB_DEPRECATE_ENABLE_JSON_MESSAGE)
#define BLPAPI_DEPRECATE_ABSTRACT_SESSION_CREATE_USER_HANDLE                  \
    BLPAPI_DEPRECATE_IMP_CONFIG(                                              \
            BLPAPI_DEPRECATE_ABSTRACT_SESSION_CREATE_USER_HANDLE_JSON)
#define BLPAPI_DEPRECATE_ABSTRACT_SESSION_CREATE_USER_HANDLE_MSG(MSG)         \
    BLPAPI_DEPRECATE_ABSTRACT_SESSION_CREATE_USER_HANDLE
#else
#define BLPAPI_DEPRECATE_ABSTRACT_SESSION_CREATE_USER_HANDLE                  \
    BLPAPI_DEPRECATE_IMP("standard",                                          \
            "https://bloomberg.github.io/blpapi-docs/cpp/3.24.4/"             \
            "classBloombergLP_1_1blpapi_1_1AbstractSession.html#"             \
            "a5e2f9d667e3f1d0bf4891268919d3e83",                              \
            "Use blpapi::AbstractSession::createIdentity() instead")
#define BLPAPI_DEPRECATE_ABSTRACT_SESSION_CREATE_USER_HANDLE_MSG(MSG)         \
    BLPAPI_DEPRECATE_IMP("standard",                                          \
            "https://bloomberg.github.io/blpapi-docs/cpp/3.24.4/"             \
            "classBloombergLP_1_1blpapi_1_1AbstractSession.html#"             \
            "a5e2f9d667e3f1d0bf4891268919d3e83",                              \
            "Use blpapi::AbstractSession::createIdentity() instead: " MSG)
#endif // defined(BB_DEPRECATE_ENABLE_JSON_MESSAGE)
#endif // defined(BLPAPI_DEPRECATE_ABSTRACT_SESSION_CREATE_USER_HANDLE__ENABLED)

#if !defined(BLPAPI_DEPRECATE_ABSTRACT_SESSION_CREATE_USER_HANDLE)
#define BLPAPI_DEPRECATE_ABSTRACT_SESSION_CREATE_USER_HANDLE
#define BLPAPI_DEPRECATE_ABSTRACT_SESSION_CREATE_USER_HANDLE_MSG(MSG)
#endif // !defined(BLPAPI_DEPRECATE_ABSTRACT_SESSION_CREATE_USER_HANDLE)

// ======================
// Macros for STRING_NAME
// ======================

#if defined(BLPAPI_DEPRECATE_STRING_NAME__ENABLED)
#error "BLPAPI_DEPRECATE_STRING_NAME__ENABLED is defined outside of the component."
#endif

#if defined(BLPAPI_DEPRECATE_ENABLE_STRING_NAME)                              \
        && defined(BLPAPI_DEPRECATE_DISABLE_STRING_NAME)
#error "Both BLPAPI_DEPRECATE_ENABLE_STRING_NAME and BLPAPI_DEPRECATE_DISABLE_STRING_NAME are enabled."
#endif

#if defined(BB_DEPRECATE_ENABLE_ALL_DEPRECATIONS_FOR_TESTING)
#define BLPAPI_DEPRECATE_STRING_NAME__ENABLED 1
#elif defined(BLPAPI_DEPRECATE_ENABLE_STRING_NAME)
#define BLPAPI_DEPRECATE_STRING_NAME__ENABLED 1
#elif (BLPAPI_DEPRECATE_STAGE_THRESHOLD                                       \
        <= BLPAPI_DEPRECATE_STAGE_LEVEL_DEFAULT_WARNING)                      \
        && (BLPAPI_DEPRECATE_SEVERITY_THRESHOLD                               \
                <= BLPAPI_DEPRECATE_SEVERITY_LEVEL_STANDARD)                  \
        && (!defined(BLPAPI_DEPRECATE_DISABLE_STRING_NAME))                   \
        && (!defined(BLPAPI_DEPRECATE_DISABLE_ALL))
#define BLPAPI_DEPRECATE_STRING_NAME__ENABLED 1
#endif // defined(BB_DEPRECATE_ENABLE_ALL_DEPRECATIONS_FOR_TESTING)

#define BLPAPI_DEPRECATE_STRING_NAME_JSON                                     \
    R"text({"feature": "string_name", "description": "Deprecating methods taking the 'name' argument by string", "severity": "standard", "stage": "default_warning", "url": "", "message": "Use the form that takes blpapi::Name instead of const char*", "library": "blpapi", "group": "blpapi"})text"

#if defined(BLPAPI_DEPRECATE_STRING_NAME__ENABLED)
#if defined(BB_DEPRECATE_ENABLE_JSON_MESSAGE)
#define BLPAPI_DEPRECATE_STRING_NAME                                          \
    BLPAPI_DEPRECATE_IMP_CONFIG(BLPAPI_DEPRECATE_STRING_NAME_JSON)
#define BLPAPI_DEPRECATE_STRING_NAME_MSG(MSG) BLPAPI_DEPRECATE_STRING_NAME
#else
#define BLPAPI_DEPRECATE_STRING_NAME                                          \
    BLPAPI_DEPRECATE_IMP("standard",                                          \
            "",                                                               \
            "Use the form that takes blpapi::Name instead of const char*")
#define BLPAPI_DEPRECATE_STRING_NAME_MSG(MSG)                                 \
    BLPAPI_DEPRECATE_IMP("standard",                                          \
            "",                                                               \
            "Use the form that takes blpapi::Name instead of const "          \
            "char*: " MSG)
#endif // defined(BB_DEPRECATE_ENABLE_JSON_MESSAGE)
#endif // defined(BLPAPI_DEPRECATE_STRING_NAME__ENABLED)

#if !defined(BLPAPI_DEPRECATE_STRING_NAME)
#define BLPAPI_DEPRECATE_STRING_NAME
#define BLPAPI_DEPRECATE_STRING_NAME_MSG(MSG)
#endif // !defined(BLPAPI_DEPRECATE_STRING_NAME)

// =============================
// Macros for PRERESOLVED_TOPICS
// =============================

#if defined(BLPAPI_DEPRECATE_PRERESOLVED_TOPICS__ENABLED)
#error "BLPAPI_DEPRECATE_PRERESOLVED_TOPICS__ENABLED is defined outside of the component."
#endif

#if defined(BLPAPI_DEPRECATE_ENABLE_PRERESOLVED_TOPICS)                       \
        && defined(BLPAPI_DEPRECATE_DISABLE_PRERESOLVED_TOPICS)
#error "Both BLPAPI_DEPRECATE_ENABLE_PRERESOLVED_TOPICS and BLPAPI_DEPRECATE_DISABLE_PRERESOLVED_TOPICS are enabled."
#endif

#if defined(BB_DEPRECATE_ENABLE_ALL_DEPRECATIONS_FOR_TESTING)
#define BLPAPI_DEPRECATE_PRERESOLVED_TOPICS__ENABLED 1
#elif defined(BLPAPI_DEPRECATE_ENABLE_PRERESOLVED_TOPICS)
#define BLPAPI_DEPRECATE_PRERESOLVED_TOPICS__ENABLED 1
#elif (BLPAPI_DEPRECATE_STAGE_THRESHOLD                                       \
        <= BLPAPI_DEPRECATE_STAGE_LEVEL_DEFAULT_WARNING)                      \
        && (BLPAPI_DEPRECATE_SEVERITY_THRESHOLD                               \
                <= BLPAPI_DEPRECATE_SEVERITY_LEVEL_STANDARD)                  \
        && (!defined(BLPAPI_DEPRECATE_DISABLE_PRERESOLVED_TOPICS))            \
        && (!defined(BLPAPI_DEPRECATE_DISABLE_ALL))
#define BLPAPI_DEPRECATE_PRERESOLVED_TOPICS__ENABLED 1
#endif // defined(BB_DEPRECATE_ENABLE_ALL_DEPRECATIONS_FOR_TESTING)

#define BLPAPI_DEPRECATE_PRERESOLVED_TOPICS_JSON                              \
    R"text({"feature": "preresolved_topics", "description": "Deprecating SubscriptionList::addResolved", "severity": "standard", "stage": "default_warning", "url": "", "message": "Deprecated since 3.25.2. Usage of pre-resolved topics is no longer supported. Usage of it should be discontinued", "library": "blpapi", "group": "blpapi"})text"

#if defined(BLPAPI_DEPRECATE_PRERESOLVED_TOPICS__ENABLED)
#if defined(BB_DEPRECATE_ENABLE_JSON_MESSAGE)
#define BLPAPI_DEPRECATE_PRERESOLVED_TOPICS                                   \
    BLPAPI_DEPRECATE_IMP_CONFIG(BLPAPI_DEPRECATE_PRERESOLVED_TOPICS_JSON)
#define BLPAPI_DEPRECATE_PRERESOLVED_TOPICS_MSG(MSG)                          \
    BLPAPI_DEPRECATE_PRERESOLVED_TOPICS
#else
#define BLPAPI_DEPRECATE_PRERESOLVED_TOPICS                                   \
    BLPAPI_DEPRECATE_IMP("standard",                                          \
            "",                                                               \
            "Deprecated since 3.25.2. Usage of pre-resolved topics is no "    \
            "longer supported. Usage of it should be discontinued")
#define BLPAPI_DEPRECATE_PRERESOLVED_TOPICS_MSG(MSG)                          \
    BLPAPI_DEPRECATE_IMP("standard",                                          \
            "",                                                               \
            "Deprecated since 3.25.2. Usage of pre-resolved topics is no "    \
            "longer supported. Usage of it should be discontinued: " MSG)
#endif // defined(BB_DEPRECATE_ENABLE_JSON_MESSAGE)
#endif // defined(BLPAPI_DEPRECATE_PRERESOLVED_TOPICS__ENABLED)

#if !defined(BLPAPI_DEPRECATE_PRERESOLVED_TOPICS)
#define BLPAPI_DEPRECATE_PRERESOLVED_TOPICS
#define BLPAPI_DEPRECATE_PRERESOLVED_TOPICS_MSG(MSG)
#endif // !defined(BLPAPI_DEPRECATE_PRERESOLVED_TOPICS)

#endif

// ----------------------------------------------------------------------------
// Copyright 2021 Bloomberg Finance L.P.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
// ----------------------------- END-OF-FILE ----------------------------------
