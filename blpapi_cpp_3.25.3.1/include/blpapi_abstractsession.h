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

/** \file blpapi_abstractsession.h */
/** \defgroup blpapi_abstractsession Component blpapi_abstractsession
\brief A common interface shared between publisher and consumer sessions.
\file blpapi_abstractsession.h
\brief A common interface shared between publisher and consumer sessions.
*/

#ifndef INCLUDED_BLPAPI_ABSTRACTSESSION
#define INCLUDED_BLPAPI_ABSTRACTSESSION

/** \addtogroup blpapi
 * @{
 */
/** \addtogroup blpapi_abstractsession
 * @{
 * <A NAME="purpose"></A>
 * <A NAME="1"> \par Purpose: </A>
 * A common interface shared between publisher and consumer sessions.
 * \par
 * \par
 * <A NAME="classes"></A>
 * <A NAME="2"> \par Classes: </A>
 * <table>
 * <tr>
 * <td>blpapi::AbstractSession</td>
 * <td>shared interface between different session types.</td>
 * </tr>
 * </table>
 * \par
 * \sa  blpapi_session, blpapi_providersession
 * \par
 * \par
 * <A NAME="description"></A>
 * <A NAME="3"> \par Description: </A>
 *  This file defines an abstract class <code>AbstractSession</code> - an
 * interface which is shared between its concrete implementations
 * <code>Session</code> and <code>ProviderSession</code>. \par
 * <A NAME="service_identifier"></A>
 * <A NAME="description.service_identifier"></A>
 * <A NAME="3.1"> \par SERVICE IDENTIFIER: </A>
 * A service identifier is the fully qualified service name which uniquely
 * identifies the service in the API infrastructure.
 * A service must be of the form <code>//\<namespace\>/\<local-name\></code>
 * where <code>\<namespace\></code> and <code>\<local-name\></code> are
 * non-empty strings of characters from the set <code>[-_.a-zA-Z0-9]</code>.
 * Service identifiers are case-insensitive, but clients are encouraged to
 * prefer identifiers without upper-case characters. Note that the
 * \<namespace\> and \<local-name\> cannot contain the character
 * <code>/</code>.
 */
/** @} */
/** @} */

#include <blpapi_authoptions.h>
#include <blpapi_call.h>
#include <blpapi_correlationid.h>
#include <blpapi_defs.h>
#include <blpapi_deprecate.h>
#include <blpapi_event.h>
#include <blpapi_eventdispatcher.h>
#include <blpapi_exception.h>
#include <blpapi_identity.h>
#include <blpapi_request.h>
#include <blpapi_service.h>
#include <blpapi_types.h>
#include <blpapi_versionmacros.h>

#ifdef __cplusplus
extern "C" {
#endif

// All of the blpapi_UserHandle_* functions have been deprecated. Please use
// blpapi_Identity_* versions of these functions instead.

BLPAPI_EXPORT
void blpapi_UserHandle_release(blpapi_UserHandle_t *handle);

BLPAPI_EXPORT
int blpapi_UserHandle_addRef(blpapi_UserHandle_t *handle);

BLPAPI_EXPORT
int blpapi_UserHandle_hasEntitlements(const blpapi_UserHandle_t *handle,
        const blpapi_Service_t *service,
        const blpapi_Element_t *eidElement,
        const int *entitlementIds,
        size_t numEntitlements,
        int *failedEntitlements,
        int *failedEntitlementsCount);

BLPAPI_EXPORT
int blpapi_AbstractSession_cancel(blpapi_AbstractSession_t *session,
        const blpapi_CorrelationId_t *correlationIds,
        size_t numCorrelationIds,
        const char *requestLabel,
        int requestLabelLen);

BLPAPI_EXPORT
int blpapi_AbstractSession_sendAuthorizationRequest(
        blpapi_AbstractSession_t *session,
        const blpapi_Request_t *request,
        blpapi_Identity_t *identity,
        blpapi_CorrelationId_t *correlationId,
        blpapi_EventQueue_t *eventQueue,
        const char *requestLabel,
        int requestLabelLen);

BLPAPI_EXPORT
int blpapi_AbstractSession_openService(
        blpapi_AbstractSession_t *session, const char *serviceIdentifier);

BLPAPI_EXPORT
int blpapi_AbstractSession_openServiceAsync(blpapi_AbstractSession_t *session,
        const char *serviceIdentifier,
        blpapi_CorrelationId_t *correlationId);

BLPAPI_EXPORT
int blpapi_AbstractSession_generateToken(blpapi_AbstractSession_t *session,
        blpapi_CorrelationId_t *correlationId,
        blpapi_EventQueue_t *eventQueue);

BLPAPI_EXPORT
int blpapi_AbstractSession_generateManualToken(
        blpapi_AbstractSession_t *session,
        blpapi_CorrelationId_t *correlationId,
        const char *user,
        const char *manualIp,
        blpapi_EventQueue_t *eventQueue);

BLPAPI_EXPORT
int blpapi_AbstractSession_getService(blpapi_AbstractSession_t *session,
        blpapi_Service_t **service,
        const char *serviceIdentifier);

BLPAPI_EXPORT
blpapi_Identity_t *blpapi_AbstractSession_createIdentity(
        blpapi_AbstractSession_t *session);

BLPAPI_EXPORT
int blpapi_AbstractSession_generateAuthorizedIdentityAsync(
        blpapi_AbstractSession_t *session,
        const blpapi_AuthOptions_t *authOptions,
        blpapi_CorrelationId_t *cid);

BLPAPI_EXPORT
int blpapi_AbstractSession_getAuthorizedIdentity(
        blpapi_AbstractSession_t *session,
        const blpapi_CorrelationId_t *cid,
        blpapi_Identity_t **identity);

BLPAPI_EXPORT
int blpapi_AbstractSession_sessionName(blpapi_AbstractSession_t *session,
        const char **sessionName,
        size_t *size);

BLPAPI_EXPORT
int blpapi_UserAgentInfo_setUserTaskName(const char *userTaskName);

BLPAPI_EXPORT
int blpapi_UserAgentInfo_setNativeSdkLanguageAndVersion(
        const char *language, const char *version);

#ifdef __cplusplus
}

#include <cassert>
#include <vector>

/** \addtogroup blpapi
 * @{
 */
/** \addtogroup blpapi_abstractsession
 * @{
 */

namespace BloombergLP {
namespace blpapi {

typedef Identity UserHandle;

// =====================
// class AbstractSession
// =====================

/*!
 *     This class provides an abstract session which defines a shared
 *     interface between publisher and consumer requests for Bloomberg.
 *
 *     Sessions manage access to services either by requests and
 *     responses or subscriptions. A Session can dispatch events and
 *     replies in either a synchronous or asynchronous mode. The mode
 *     of a Session is determined when it is constructed and cannot be
 *     changed subsequently.
 *
 *     A Session is asynchronous if an EventHandler object is
 *     supplied when it is constructed. The setEventHandler() method
 *     may be called to adjust the way events are handled subsequently
 *     and the nextEvent() method may not be called. All incoming
 *     events are delivered to the EventHandler(s) supplied on
 *     construction or subsequently using setEventHandler().
 *
 *     A Session is synchronous if an EventHandler object is not
 *     supplied when it is constructed. The nextEvent() method must be
 *     called to read incoming events and the setEventHandler() method
 *     may not be called.
 *
 *     Several methods in Session take a CorrelationId parameter. The
 *     application may choose to supply its own CorrelationId values
 *     or allow the Session to create values. If the application
 *     supplies its own CorrelationId values it must manage their
 *     lifetime such that the same value is not reused for more than
 *     one operation at a time. The lifetime of a CorrelationId begins
 *     when it is supplied in a method invoked on a Session and ends
 *     either when it is explicitly cancelled using cancel() or
 *     unsubscribe(), when a RESPONSE Event (not a PARTIAL_RESPONSE)
 *     containing it is received or when a SUBSCRIPTION_STATUS Event
 *     which indicates that the subscription it refers to has been
 *     terminated is received.
 *
 *     When using an asynchronous Session the application must be
 *     aware that because the callbacks are generated from another
 *     thread they may be processed before the call which generates
 *     them has returned. For example, the SESSION_STATUS Event
 *     generated by a startAsync() may be processed before
 *     startAsync() has returned (even though startAsync() itself will
 *     not block).
 *
 *     This becomes more significant when Session generated
 *     CorrelationIds are in use. For example, if a call to
 *     subscribe() which returns a Session generated CorrelationId has
 *     not completed before the first Events which contain that
 *     CorrelationId arrive the application may not be able to
 *     interpret those events correctly. For this reason, it is
 *     preferable to use user generated CorrelationIds when using
 *     asynchronous Sessions. This issue does not arise when using a
 *     synchronous Session as long as the calls to subscribe() etc are
 *     made on the same thread as the calls to nextEvent().
 */
/*!
 * See \ref blpapi_abstractsession
 */
class AbstractSession {

    blpapi_AbstractSession_t *d_handle_p;

  private:
    // NOT IMPLEMENTED
    AbstractSession(const AbstractSession&);
    AbstractSession& operator=(const AbstractSession&);

  protected:
    //! \cond INTERNAL
    AbstractSession();
    /*!<
     *     Create an abstract session object.
     */

    void initAbstractSessionHandle(blpapi_AbstractSession_t *handle);
    /*!<
     *     Initialize the handle of this abstract session.
     */
    //! \endcond

  public:
    virtual ~AbstractSession();
    /*!<
     *     Destructor.
     */

    // MANIPULATORS
    virtual bool start() = 0;
    /*!<
     *     Attempt to start this Session and blocks until the Session
     *     has started or failed to start. If the Session is started
     *     successfully <code>true</code> is returned, otherwise
     *     <code>false</code> is returned. Before start() returns a
     *     SESSION_STATUS Event is generated. If this is an asynchronous
     *     Session then the SESSION_STATUS may be processed by the registered
     *     EventHandler before start() has returned. A Session may
     *     only be started once.
     */

    virtual bool startAsync() = 0;
    /*!<
     *     Attempt to begin the process to start this Session and
     *     return <code>true</code> if successful, otherwise return
     *     <code>false</code>. The application must monitor events for a
     *     SESSION_STATUS Event which will be generated once the Session has
     *     started or if it fails to start. If this is an asynchronous Session
     *     then the SESSION_STATUS Event may be processed by the registered
     *     EventHandler before startAsync() has returned. A Session may
     *     only be started once.
     */

    virtual void stop() = 0;
    /*!<
     *     Stop operation of this session and block until all callbacks to
     *     EventHandler objects relating to this Session which are currently in
     *     progress have completed (including the callback to handle the
     *     SESSION_STATUS Event with SessionTerminated message this call
     *     generates). Once this returns no further callbacks to EventHandlers
     *     will occur. If stop() is called from within an EventHandler callback
     *     the behavior is undefined and may result in a deadlock. Once a
     *     Session has been stopped it can only be destroyed.
     */

    virtual void stopAsync() = 0;
    /*!<
     *     Begin the process to stop this Session and return immediately. The
     *     application must monitor events for a SESSION_STATUS Event with
     *     SessionTerminated message which will be generated once the
     *     Session has been stopped. After this SESSION_STATUS Event no further
     *     callbacks to EventHandlers will occur. This method can be called
     *     from within an EventHandler callback to stop Sessions using
     *     non-default (external) EventDispatcher. Once a Session has been
     *     stopped it can only be destroyed.
     */

    virtual Event nextEvent(int timeout = 0) = 0;
    /*!<
     *     Return the next available Event for this session. If there
     *     is no event available this will block for up to the
     *     specified <code>timeout</code> in milliseconds for an Event to
     *     arrive. A value of 0 for <code>timeout</code> (the default)
     *     indicates nextEvent() should not timeout and will not
     *     return until the next Event is available.
     *
     *     If nextEvent() returns due to a timeout it will return an
     *     event of type <code>EventType::TIMEOUT</code>.
     *
     *     If this is invoked on a Session which was created in
     *     asynchronous mode an InvalidStateException is thrown.
     */

    virtual int tryNextEvent(Event *event) = 0;
    /*!<
     *     If there are Events available for the session, load the next Event
     *     into event and return 0 indicating success. If there is no event
     *     available for the session, return a non-zero value with no effect
     *     on event. This method never blocks.
     */

    virtual bool openService(const char *serviceIdentifier);
    /*!<
     *     Attempt to open the service identified by the specified
     *     <code>serviceIdentifier</code> and block until the service is either
     *     opened successfully or has failed to be opened. Return
     *     <code>true</code> if the service is opened successfully and
     *     <code>false</code> if the service cannot be successfully opened.
     *
     *     The <code>serviceIdentifier</code> must contain a fully qualified
     *     service name. That is, it must be of the form
     *     <code>//\<namespace\>/\<local-name\></code>.
     *
     *     Before openService() returns a SERVICE_STATUS Event is generated. If
     *     this is an asynchronous Session then this
     *     Event may be processed by the registered EventHandler
     *     before openService() has returned.
     */

    virtual CorrelationId openServiceAsync(const char *serviceIdentifier,
            const CorrelationId& correlationId = CorrelationId());
    /*!<
     *     Begin the process to open the service identified by the specified
     *     <code>serviceIdentifier</code> and return immediately. The optional
     *     specified <code>correlationId</code> is used to track Events
     *     generated as a result of this call. The actual correlationId which
     *     will identify Events generated as a result of this call is returned.
     *
     *     The <code>serviceIdentifier</code> must contain a fully qualified
     *     service name. That is, it must be of the form
     *     <code>//\<namespace\>/\<local-name\></code>.
     *
     *     The application must monitor events for a SERVICE_STATUS
     *     Event which will be generated once the service has been
     *     successfully opened or the opening has failed.
     */

    virtual CorrelationId sendAuthorizationRequest(
            const Request& authorizationRequest,
            Identity *identity,
            const CorrelationId& correlationId = CorrelationId(),
            EventQueue *eventQueue = 0);
    /*!<
     *     Send the specified <code>authorizationRequest</code> and update the
     *     specified <code>identity</code> with the results. If the optionally
     *     specified <code>correlationId</code> is supplied, it is used;
     *     otherwise create a CorrelationId. The actual CorrelationId used is
     *     returned.  If the optionally specified <code>eventQueue</code> is
     *     supplied all
     *     Events relating to this Request will arrive on that EventQueue.
     *
     *     A successful request will generate zero or more PARTIAL_RESPONSE
     *     Messages followed by exactly one RESPONSE Message. Once the final
     *     RESPONSE Message has been received the specified
     *     <code>identity</code> will have been updated to contain the users
     *     entitlement information and the CorrelationId associated with the
     *     request may be re-used. If the request fails at any stage a
     *     REQUEST_STATUS will be generated, the specified
     *     <code>identity</code> will not be modified and the CorrelationId may
     *     be re-used.
     *
     *     The <code>identity</code> supplied must have been returned from this
     *     Session's <code>createIdentity()</code> method. For example
     *     @code
     *     Identity handle(session.createIdentity());
     *     session.sendAuthorizationRequest(authRequest, &handle, ...)
     *     @endcode
     */

    virtual void cancel(const CorrelationId& correlationId);
    /*!<
     *     If the specified <code>correlationId</code> identifies a current
     *     request then cancel that request.
     *
     *     Once this call returns the specified <code>correlationId</code> will
     *     not be seen in any subsequent Message obtained from a
     *     MessageIterator by calling next(). However, any Message
     *     currently pointed to by a MessageIterator when
     *     cancel() is called is not affected even if it has the
     *     specified <code>correlationId</code>. Also any Message where a
     *     reference has been retained by the application may still
     *     contain the <code>correlationId</code>. For these reasons, although
     *     technically an application is free to re-use
     *     <code>correlationId</code> as soon as this method returns it is
     *     preferable not to aggressively re-use correlation IDs,
     *     particularly with an asynchronous Session.
     */

    virtual void cancel(const std::vector<CorrelationId>& correlationIds);
    /*!<
     *     For each value in the specified <code>correlationIds</code> which
     *     identifies a current request then cancel that request. Any
     *     values in the specified <code>correlationIds</code> which do not
     *     identify a current request are ignored.
     *
     *     Once this call returns the specified <code>correlationIds</code>
     *     will not be seen in any subsequent Message obtained from a
     *     MessageIterator by calling next(). However, any Message
     *     currently pointed to by a MessageIterator when
     *     cancel() is called is not affected even if it has one
     *     of the specified <code>correlationIds</code>. Also any Message where
     *     a reference has been retained by the application may still contain
     *     one of the <code>correlationIds</code>. For these reasons, although
     *     technically an application is free to re-use any of the
     *     <code>correlationIds</code> as soon as this method returns it is
     *     preferable not to aggressively re-use correlation IDs, particularly
     *     with an asynchronous Session.
     */

    virtual void cancel(
            const CorrelationId *correlationIds, size_t numCorrelationIds);
    /*!<
     *     For each value specified <code>correlationIds</code> and
     *     <code>numCorrelationIds</code> which identifies a current request
     *     then cancel that request. Any specified CorrelationId's which do not
     *     identify a current request are ignored.
     *
     *     Once this call returns the specified <code>correlationIds</code>
     *     will not be seen in any subsequent Message obtained from a
     *     MessageIterator by calling next(). However, any Message
     *     currently pointed to by a MessageIterator when
     *     cancel() is called is not affected even if it has one
     *     of the specified <code>correlationIds</code>. Also any Message where
     *     a reference has been retained by the application may still contain
     *     one of the <code>correlationIds</code>. For these reasons, although
     *     technically an application is free to re-use any of the
     *     <code>correlationIds</code> as soon as this method returns it is
     *     preferable not to aggressively re-use correlation IDs, particularly
     *     with an asynchronous Session.
     */

    virtual CorrelationId generateToken(
            const CorrelationId& correlationId = CorrelationId(),
            EventQueue *eventQueue = 0);
    /*!<
     *     Generate a token to be used for authorization.
     *     If an invalid authentication option is specified in session option
     *     or there is a failure to get authentication information based on
     *     authentication option, or if the authentication mode is
     *     <code>MANUAL</code> for a user or user and application
     *     authentication, then an InvalidArgumentException is thrown.
     */

    virtual CorrelationId generateToken(const char *userId,
            const char *ipAddress,
            const CorrelationId& correlationId = CorrelationId(),
            EventQueue *eventQueue = 0);
    /*!<
     *     Generate a token to be used for authorization, using the specified
     *     <code>userId</code> and <code>ipAddress</code>.  If the
     *     authentication mode is not <code>MANUAL</code>, if the
     *     <code>userId</code> or <code>ipAddress</code> are not valid or if
     *     there's a problem obtaining the authentication information based on
     *     the authentication options in <code>SessionOptions</code>, then an
     *     InvalidArgumentException is thrown.
     */

    virtual Service getService(const char *serviceIdentifier) const;
    /*!<
     *     Return a Service object representing the service
     *     identified by the specified <code>serviceIdentifier</code>
     *
     *     The <code>serviceIdentifier</code> must contain a fully qualified
     *     service name. That is, it must be of the form
     *     <code>//\<namespace\>/\<local-name\></code>.
     *
     *     If the service identified by <code>serviceIdentifier</code> is not
     *     open or registered already then a <code>NotFoundException</code> is
     *     thrown.
     */

    virtual Identity createIdentity();
    /*!<
     *     Return a Identity which is valid but has not been
     *     authorized.
     */

    CorrelationId generateAuthorizedIdentity(const AuthOptions& authOptions,
            const CorrelationId& cid = CorrelationId());
    /*!<
     *     Generates an authorized <code>Identity</code> with the specified
     *     <code>AuthOptions</code> and optionally specified <code>cid</code>.
     *
     *     The optionally specified <code>cid</code> is used to track
     *     <code>Event</code> objects generated as a result of this call.
     *     Return the actual <code>CorrelationId</code> object that will
     *     identify the messages associated with the generated identity.
     *
     *     One or more <code>AUTHORIZATION_STATUS</code> events, zero or more
     *     <code>TOKEN_STATUS</code> events and zero or more
     *     <code>SERVICE_STATUS</code> events are generated.  If this is an
     *     asynchronous session then these <code>Event</code>s may be processed
     *     by the registered <code>EventHandler</code> before
     *     <code>generateAuthorizedIdentity()</code> has returned.
     */

    Identity getAuthorizedIdentity(
            const CorrelationId& correlationId = CorrelationId());
    /*!<
     *     Returns the authorized <code>Identity</code> associated with
     *     <code>correlationId</code>
     *
     *     A <code>NotFoundException</code> is thrown if there is no
     *     <code>Identity</code> associated with <code>correlationId</code>, if
     *     the associated <code>Identity</code> is not authorized, or if
     *     <code>correlationId</code> is not given and the session
     *     <code>Identity</code> is not authorized.
     */

    std::string sessionName();
    /*!<
     *     Returns the session name.
     */

    //! \cond INTERNAL
    blpapi_AbstractSession_t *abstractSessionHandle() const;
    /*!<
     *     Return the handle of this abstract session.
     */
    //! \endcond

    //! \name DEPRECATED
    BLPAPI_DEPRECATE_ABSTRACT_SESSION_CREATE_USER_HANDLE UserHandle
    createUserHandle();
    /*!<
     *     \deprecated Use createIdentity() instead.
     *
     *     Return a UserHandle which is valid but has not been
     *     authorized.
     */
};

/** @} */
/** @} */

//=============================================================================
//                           INLINE FUNCTION DEFINITIONS
//=============================================================================

// ---------------------
// class AbstractSession
// ---------------------

inline AbstractSession::AbstractSession()
    : d_handle_p(0)
{
    // Ignore the return value
    BLPAPI_CALL(blpapi_UserAgentInfo_setNativeSdkLanguageAndVersion)
    ("C++", BLPAPI_SDK_VERSION_STRING);
}

inline AbstractSession::~AbstractSession() { }

inline void AbstractSession::initAbstractSessionHandle(
        blpapi_AbstractSession_t *handle)
{
    d_handle_p = handle;
}

inline Service AbstractSession::getService(const char *serviceIdentifier) const
{
    blpapi_Service_t *service;
    ExceptionUtil::throwOnError(blpapi_AbstractSession_getService(
            d_handle_p, &service, serviceIdentifier));
    return Service(service);
}

inline CorrelationId AbstractSession::sendAuthorizationRequest(
        const Request& authorizationRequest,
        Identity *identity,
        const CorrelationId& correlationId,
        EventQueue *eventQueue)
{
    assert(identity);

    CorrelationId retCorrelationId(correlationId);

    ExceptionUtil::throwOnError(
            blpapi_AbstractSession_sendAuthorizationRequest(d_handle_p,
                    authorizationRequest.handle(),
                    identity->handle(),
                    const_cast<blpapi_CorrelationId_t *>(
                            &retCorrelationId.impl()),
                    eventQueue ? eventQueue->handle() : 0,
                    0,
                    0));

    return retCorrelationId;
}

inline void AbstractSession::cancel(const CorrelationId& correlationId)
{
    blpapi_AbstractSession_cancel(d_handle_p, &correlationId.impl(), 1, 0, 0);
}

inline void AbstractSession::cancel(
        const std::vector<CorrelationId>& correlationIds)
{
    if (!correlationIds.size()) {
        return;
    }
    cancel(&correlationIds[0], correlationIds.size());
}

inline void AbstractSession::cancel(
        const CorrelationId *correlationIds, size_t numCorrelationIds)
{
    blpapi_AbstractSession_cancel(d_handle_p,
            reinterpret_cast<const blpapi_CorrelationId_t *>(correlationIds),
            numCorrelationIds,
            0,
            0);
}

inline CorrelationId AbstractSession::generateToken(
        const CorrelationId& correlationId, EventQueue *eventQueue)
{
    CorrelationId retCorrelationId(correlationId);

    ExceptionUtil::throwOnError(
            blpapi_AbstractSession_generateToken(d_handle_p,
                    &retCorrelationId.impl(),
                    eventQueue ? eventQueue->handle() : 0));

    return retCorrelationId;
}

inline CorrelationId AbstractSession::generateToken(const char *userId,
        const char *ipAddress,
        const CorrelationId& correlationId,
        EventQueue *eventQueue)
{
    CorrelationId retCorrelationId(correlationId);

    ExceptionUtil::throwOnError(
            BLPAPI_CALL(blpapi_AbstractSession_generateManualToken)(d_handle_p,
                    &retCorrelationId.impl(),
                    userId,
                    ipAddress,
                    eventQueue ? eventQueue->handle() : 0));

    return retCorrelationId;
}

inline bool AbstractSession::openService(const char *serviceIdentifier)
{
    return blpapi_AbstractSession_openService(d_handle_p, serviceIdentifier)
            ? false
            : true;
}

inline CorrelationId AbstractSession::openServiceAsync(
        const char *serviceIdentifier, const CorrelationId& correlationId)
{
    blpapi_CorrelationId_t retv = correlationId.impl();
    ExceptionUtil::throwOnError(blpapi_AbstractSession_openServiceAsync(
            d_handle_p, serviceIdentifier, &retv));

    return CorrelationId(retv);
}

inline UserHandle AbstractSession::createUserHandle()
{
    return UserHandle(blpapi_AbstractSession_createIdentity(d_handle_p));
}

inline Identity AbstractSession::createIdentity()
{
    return Identity(blpapi_AbstractSession_createIdentity(d_handle_p));
}

inline CorrelationId AbstractSession::generateAuthorizedIdentity(
        const AuthOptions& authOptions, const CorrelationId& cid)
{
    blpapi_CorrelationId_t cidHandle = cid.impl();

    ExceptionUtil::throwOnError(BLPAPI_CALL(
            blpapi_AbstractSession_generateAuthorizedIdentityAsync)(
            d_handle_p, authOptions.handle(), &cidHandle));

    return CorrelationId(cidHandle);
}

inline Identity AbstractSession::getAuthorizedIdentity(
        const CorrelationId& correlationId)
{
    blpapi_Identity_t *identity = NULL;

    ExceptionUtil::throwOnError(
            BLPAPI_CALL(blpapi_AbstractSession_getAuthorizedIdentity)(
                    d_handle_p, &correlationId.impl(), &identity));

    return Identity(identity);
}

inline std::string AbstractSession::sessionName()
{
    const char *sessionName = nullptr;
    size_t size = 0;

    ExceptionUtil::throwOnError(
            BLPAPI_CALL(blpapi_AbstractSession_sessionName)(
                    d_handle_p, &sessionName, &size));
    return std::string(sessionName, size);
}

inline blpapi_AbstractSession_t *AbstractSession::abstractSessionHandle() const
{
    return d_handle_p;
}

} // close namespace blpapi
} // close namespace BloombergLP

#endif // #ifdef __cplusplus
#endif // #ifndef INCLUDED_BLPAPI_ABSTRACTSESSION
