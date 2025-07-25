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

/** \file blpapi_message.h */
/** \defgroup blpapi_message Component blpapi_message
\brief Defines a message containing elements.
\file blpapi_message.h
\brief Defines a message containing elements.
*/

#ifndef INCLUDED_BLPAPI_MESSAGE
#define INCLUDED_BLPAPI_MESSAGE

/** \addtogroup blpapi
 * @{
 */
/** \addtogroup blpapi_message
 * @{
 * <A NAME="purpose"></A>
 * <A NAME="1"> \par Purpose: </A>
 * Defines a message containing elements.
 * \par
 * \par
 * <A NAME="classes"></A>
 * <A NAME="2"> \par Classes: </A>
 * <table>
 * <tr>
 * <td>blpapi::Message</td>
 * <td>individual message inside an event.</td>
 * </tr>
 * </table>
 * \par
 * \sa  blpapi_event
 * \par
 * \par
 * <A NAME="description"></A>
 * <A NAME="3"> \par Description: </A>
 *  This file defines a <code>Message</code>. A <code>Message</code> is
 * contained in an <code>Event</code> and contains <code>Element</code>s.
 */
/** @} */
/** @} */

#include <blpapi_call.h>
#include <blpapi_correlationid.h>
#include <blpapi_defs.h>
#include <blpapi_deprecate.h>
#include <blpapi_element.h>
#include <blpapi_name.h>
#include <blpapi_service.h>
#include <blpapi_streamproxy.h>
#include <blpapi_timepoint.h>

struct blpapi_Message;
typedef struct blpapi_Message blpapi_Message_t;

#ifdef __cplusplus
extern "C" {
#endif

BLPAPI_EXPORT
blpapi_Name_t *blpapi_Message_messageType(const blpapi_Message_t *message);

BLPAPI_EXPORT
const char *blpapi_Message_typeString(const blpapi_Message_t *message);

BLPAPI_EXPORT
const char *blpapi_Message_topicName(const blpapi_Message_t *message);

BLPAPI_EXPORT
blpapi_Service_t *blpapi_Message_service(const blpapi_Message_t *message);

BLPAPI_EXPORT
int blpapi_Message_numCorrelationIds(const blpapi_Message_t *message);

BLPAPI_EXPORT
blpapi_CorrelationId_t blpapi_Message_correlationId(
        const blpapi_Message_t *message, size_t index);

BLPAPI_EXPORT
int blpapi_Message_getRequestId(
        const blpapi_Message_t *message, const char **requestId);

BLPAPI_EXPORT
blpapi_Element_t *blpapi_Message_elements(const blpapi_Message_t *message);

BLPAPI_EXPORT
const char *blpapi_Message_privateData(
        const blpapi_Message_t *message, size_t *size);

BLPAPI_EXPORT
int blpapi_Message_fragmentType(const blpapi_Message_t *message);

BLPAPI_EXPORT
int blpapi_Message_recapType(const blpapi_Message_t *message);

BLPAPI_EXPORT
int blpapi_Message_print(const blpapi_Message_t *message,
        blpapi_StreamWriter_t streamWriter,
        void *stream,
        int indentLevel,
        int spacesPerLevel);

BLPAPI_EXPORT
int blpapi_Message_addRef(const blpapi_Message_t *message);

BLPAPI_EXPORT
int blpapi_Message_release(const blpapi_Message_t *message);

BLPAPI_EXPORT
int blpapi_Message_timeReceived(
        const blpapi_Message_t *message, blpapi_TimePoint_t *timeReceived);

#ifdef __cplusplus
}

/** \addtogroup blpapi
 * @{
 */
/** \addtogroup blpapi_message
 * @{
 */

namespace BloombergLP {
namespace blpapi {

/*!
 *     A handle to a single message.
 *
 *     Message objects are obtained from a MessageIterator. Each
 *     Message is associated with a Service and with one or more
 *     CorrelationId values. The Message contents are represented as
 *     an Element and some convenient shortcuts are supplied to the
 *     Element accessors.
 *
 *     A Message is a handle to a single underlying protocol
 *     message. The underlying messages are reference counted - the
 *     underlying Message object is freed when the last Message object
 *     which references it is destroyed.
 */
/*!
 * See \ref blpapi_message
 */
class Message {

    blpapi_Message_t *d_handle;
    Element d_elements;
    bool d_isCloned;

  public:
    /// A message could be split into more than one fragments to reduce
    /// each message size. This enum is used to indicate whether a message
    /// is a fragmented message and the position in the fragmented messages.
    enum Fragment {

        FRAGMENT_NONE = BLPAPI_MESSAGE_FRAGMENT_NONE,
        ///< message is not fragmented
        FRAGMENT_START = BLPAPI_MESSAGE_FRAGMENT_START,
        ///< the first fragmented message
        FRAGMENT_INTERMEDIATE = BLPAPI_MESSAGE_FRAGMENT_INTERMEDIATE,
        ///< intermediate fragmented messages
        FRAGMENT_END = BLPAPI_MESSAGE_FRAGMENT_END
        ///< the last fragmented message
    };

    /*!
     * Some subscription data messages (messages which originate from events
     * with event type <code>blpapi::Event::SUBSCRIPTION_DATA</code>) are
     * recaps, which summarize the overall state of the data associated with
     * the subscription. A solicited recap message is a recap intended for a
     * particular consumer who has recently subscribed or resubscribed to a
     * topic on a service. Services may also emit unsolicited recaps, intended
     * for all consumers, at any time. This enum is used to indicate the recap
     * type for a particular subscription data message.
     */
    /*!
     * See \ref blpapi_message
     */
    struct RecapType {
        enum Type {
            e_none = BLPAPI_MESSAGE_RECAPTYPE_NONE,
            ///< normal data tick; not a recap
            e_solicited = BLPAPI_MESSAGE_RECAPTYPE_SOLICITED,
            ///< generated on request by subscriber
            e_unsolicited = BLPAPI_MESSAGE_RECAPTYPE_UNSOLICITED
            ///< generated at discretion of the service
        };
    };

  public:
    // CREATORS

    //! \cond INTERNAL
    explicit Message(blpapi_Message_t *handle, bool clonable = false);
    /*!<
     * Construct the Message with the specified <code>handle</code> and set the
     * isCloned flag with the specified value of <code>clonable</code>.
     *
     * This flag will used to release reference on message handle when the
     * destructor is called.
     */
    //! \endcond

    Message(const Message& original);
    /*!<
     * Construct the message using the handle of the original.
     *
     * \internal This will add a reference to the handle and set the d_isCloned
     * flag to true, to ensure that release reference is called when destructor
     * is invoked.
     */

    ~Message();
    /*!<
     * Destroy this message.
     *
     * \internal Call release reference on handle if the d_isCloned is set.
     */

    // MANIUPLATORS
    Message& operator=(const Message& rhs);
    /*!<
     * Copies the message specified by <code>rhs</code> into the current
     * message.
     *
     * \internal Also set the `d_isCloned` flag to `true`. This flag will used
     * to release reference on message handle when the destructor is called.
     */

    // ACCESSORS
    Name messageType() const;
    /*!<
     * Returns the type of this message.
     */

    BLPAPI_DEPRECATE_MESSAGE_TOPIC_NAME const char *topicName() const;
    /*!<
     * \deprecated This method always returns an empty string.
     *
     * Returns a pointer to a null-terminated string containing the topic
     * string associated with this message. If there is no topic associated
     * with this message then an empty string is returned. The pointer returned
     * remains valid until the Message is destroyed.
     *
     * This function has been deprecated because messages could contain
     * multiple payloads with different correlation ids, and each of these
     * correlation ids may map to different topic strings.
     *
     * In such scenario, it will be incorrect to return one out of the
     * topics(for the various correlation ids in the message) as the topic name
     * for the message.Trying to make this correct will result in some extra
     * look up costs.
     *
     * For correctness, users are encouraged to maintain a data structure in
     * their application to help retrieve the topic name associated with the
     * cid's present in the delivered message.
     */

    Service service() const;
    /*!<
     * Returns the service which sent this Message.
     */

    int numCorrelationIds() const;
    /*!<
     * Returns the number of CorrelationIds associated with this message.
     *
     * Note: A Message will have exactly one CorrelationId unless
     * <code>allowMultipleCorrelatorsPerMsg</code> option was enabled for the
     * Session this Message came from. When
     * <code>allowMultipleCorrelatorsPerMsg</code> is disabled (the default)
     * and more than one active subscription would result in the same Message
     * the Message is delivered multiple times (without making physical
     * copied). Each Message is accompanied by a single CorrelationId. When
     * <code>allowMultipleCorrelatorsPerMsg</code> is enabled and more than one
     * active subscription would result in the same Message the Message is
     * delivered once with a list of corresponding CorrelationId values.
     */

    CorrelationId correlationId(size_t index = 0) const;
    /*!<
     * Returns the specified <code>index</code>th CorrelationId associated with
     * this message. If <code>index</code>\>=numCorrelationIds() then an
     * exception is thrown.
     */

    RecapType::Type recapType() const;
    /*!<
     * Return the recap type of this message. The return value is a value of
     * enum <code>RecapType::Type</code> to indicate whether this message is a
     * solicited recap, an unsolicited recap, or neither.
     */

    bool hasElement(const Name& name, bool excludeNullElements = false) const;
    /*!<
     * Equivalent to asElement().hasElement(name).
     */

    BLPAPI_DEPRECATE_STRING_NAME bool hasElement(
            const char *name, bool excludeNullElements = false) const;
    /*!<
     * \deprecated Use the form that takes Name instead of const char* or
     * `hasElement(Name::findName(name))` when fields are not known ahead
     * of time.
     *
     * Equivalent to asElement().hasElement(Name::findName(name)).
     */

    size_t numElements() const;
    /*!<
     * Equivalent to asElement().numElements().
     */

    const Element getElement(const Name& name) const;
    /*!<
     * Equivalent to asElement().getElement(name).
     */

    BLPAPI_DEPRECATE_STRING_NAME const Element getElement(
            const char *name) const;
    /*!<
     * \deprecated Use the form that takes Name instead of const char*.
     *
     * Equivalent to asElement().getElement(name).
     */

    bool getElementAsBool(const Name& name) const;
    /*!<
     * Equivalent to asElement().getElementAsBool(name).
     */

    BLPAPI_DEPRECATE_STRING_NAME bool getElementAsBool(const char *name) const;
    /*!<
     * \deprecated Use the form that takes Name instead of const char*.
     *
     * Equivalent to asElement().getElementAsBool(name).
     */

    char getElementAsChar(const Name& name) const;
    /*!<
     * Equivalent to asElement().getElementAsChar(name).
     */

    BLPAPI_DEPRECATE_STRING_NAME char getElementAsChar(const char *name) const;
    /*!<
     * \deprecated Use the form that takes Name instead of const char*.
     *
     * Equivalent to asElement().getElementAsChar(name).
     */

    Int32 getElementAsInt32(const Name& name) const;
    /*!<
     * Equivalent to asElement().getElementAsInt32(name).
     */

    BLPAPI_DEPRECATE_STRING_NAME Int32 getElementAsInt32(
            const char *name) const;
    /*!<
     * \deprecated Use the form that takes Name instead of const char*.
     *
     * Equivalent to asElement().getElementAsInt32(name).
     */

    Int64 getElementAsInt64(const Name& name) const;
    /*!<
     * Equivalent to asElement().getElementAsInt64(name).
     */

    BLPAPI_DEPRECATE_STRING_NAME Int64 getElementAsInt64(
            const char *name) const;
    /*!<
     * \deprecated Use the form that takes Name instead of const char*.
     *
     * Equivalent to asElement().getElementAsInt64(name).
     */

    Float32 getElementAsFloat32(const Name& name) const;
    /*!<
     * Equivalent to asElement().getElementAsFloat32(name).
     */

    BLPAPI_DEPRECATE_STRING_NAME Float32 getElementAsFloat32(
            const char *name) const;
    /*!<
     * \deprecated Use the form that takes Name instead of const char*.
     *
     * Equivalent to asElement().getElementAsFloat32(name).
     */

    Float64 getElementAsFloat64(const Name& name) const;
    /*!<
     * Equivalent to asElement().getElementAsFloat64(name).
     */

    BLPAPI_DEPRECATE_STRING_NAME Float64 getElementAsFloat64(
            const char *name) const;
    /*!<
     * \deprecated Use the form that takes Name instead of const char*.
     *
     * Equivalent to asElement().getElementAsFloat64(name).
     */

    Datetime getElementAsDatetime(const Name& name) const;
    /*!<
     * Equivalent to asElement().getElementAsDatetime(name).
     */

    BLPAPI_DEPRECATE_STRING_NAME Datetime getElementAsDatetime(
            const char *name) const;
    /*!<
     * \deprecated Use the form that takes Name instead of const char*.
     *
     * Equivalent to asElement().getElementAsDatetime(name).
     */

    const char *getElementAsString(const Name& name) const;
    /*!<
     * Equivalent to asElement().getElementAsString(name).
     */

    BLPAPI_DEPRECATE_STRING_NAME const char *getElementAsString(
            const char *name) const;
    /*!<
     * \deprecated Use the form that takes Name instead of const char*.
     *
     * Equivalent to asElement().getElementAsString(name).
     */

    Bytes getElementAsBytes(const Name& name) const;
    /*!<
     * Equivalent to asElement().getElementAsBytes(name).
     */

    const char *getRequestId() const;
    /*!<
     * Return a pointer offering unmodifiable access to the message's request
     * id if one exists, otherwise return null.
     *
     * When present, the request id can be reported to Bloomberg to
     * troubleshoot the cause of failure messages, or issues with the data
     * contained in the message.
     *
     * Note that request id is not the same as correlation id and should not be
     * used for correlation purposes.
     */

    const Element asElement() const;
    /*!<
     * Returns the contents of this Message as a read-only Element. The Element
     * returned remains valid until this Message is destroyed.
     */

    BLPAPI_DEPRECATE_STRING_NAME const char *getPrivateData(
            size_t *size) const;
    /*!<
     * \deprecated This method is no longer supported.
     *
     * Return a raw pointer to the message private data if it had any. If
     * <code>size</code> is a valid pointer (not 0), it will be filled with the
     * size of the private data. If the message has no private data attached to
     * it the return value is 0 and the <code>size</code> pointer (if valid) is
     * set to 0.
     */

    Fragment fragmentType() const;
    /*!<
     * Return fragment type of this message. The return value is a value of
     * enum Fragment to indicate whether it is a fragmented message of a big
     * message and its positions in fragmentation if it is.
     */

    int timeReceived(TimePoint *timestamp) const;
    /*!<
     * Load into the specified <code>timestamp</code>, the time when the
     * message was received by the sdk. This method will fail if there is no
     * timestamp associated with Message. On failure, the
     * <code>timestamp</code> is not modified. Return 0 on success and a
     * non-zero value otherwise. Note that by default the subscription data
     * messages are not timestamped (but all the other messages are). To enable
     * recording receive time for subscription data, set
     * <code>SessionOptions::recordSubscriptionDataReceiveTimes</code>.
     */

    std::ostream& print(
            std::ostream& stream, int level = 0, int spacesPerLevel = 4) const;
    /*!<
     * Format this Message to the specified output <code>stream</code> at the
     * (absolute value of) the optionally specified indentation
     * <code>level</code> and return a reference to <code>stream</code>. If
     * <code>level</code> is specified, optionally specify
     * <code>spacesPerLevel</code>, the number of spaces per indentation level
     * for this and all of its nested objects. If <code>level</code> is
     * negative, suppress indentation of the first line. If
     * <code>spacesPerLevel</code> is negative, format the entire output on one
     * line, suppressing all but the initial indentation (as governed by
     * <code>level</code>).
     */

    //! \cond INTERNAL
    const blpapi_Message_t *impl() const;
    /*!<
     * Returns the internal implementation.
     */

    blpapi_Message_t *impl();
    /*!<
     * Returns the internal implementation.
     */
    //! \endcond
};

/** @} */
/** @} */

// FREE OPERATORS
std::ostream& operator<<(std::ostream& stream, const Message& message);
/*!<
 * Write the value of the specified <code>message</code> object to the
 * specified output <code>stream</code> in a single-line format, and return a
 * reference to <code>stream</code>.  If <code>stream</code> is not valid on
 * entry, this operation has no effect.  Note that this human-readable format
 * is not fully specified, can change without notice, and is logically
 * equivalent to:
\code
  print(stream, 0, -1);
\endcode
 */

// ============================================================================
//                      INLINE AND TEMPLATE FUNCTION IMPLEMENTATIONS
// ============================================================================

// -------------
// class Message
// -------------
// CREATORS
inline Message::Message(blpapi_Message_t *handle, bool clonable)
    : d_handle(handle)
    , d_isCloned(clonable)
{
    if (handle) {
        d_elements = Element(blpapi_Message_elements(handle));
    }
}

inline Message::Message(const Message& original)
    : d_handle(original.d_handle)
    , d_elements(original.d_elements)
    , d_isCloned(true)
{
    if (d_handle) {
        BLPAPI_CALL_MESSAGE_ADDREF(d_handle);
    }
}

inline Message::~Message()
{
    if (d_isCloned && d_handle) {
        BLPAPI_CALL_MESSAGE_RELEASE(d_handle);
    }
}
// MANIPULATORS
inline Message& Message::operator=(const Message& rhs)
{

    if (this == &rhs) {
        return *this;
    }

    if (d_isCloned && (d_handle == rhs.d_handle)) {
        return *this;
    }

    if (d_isCloned && d_handle) {
        BLPAPI_CALL_MESSAGE_RELEASE(d_handle);
    }
    d_handle = rhs.d_handle;
    d_elements = rhs.d_elements;
    d_isCloned = true;

    if (d_handle) {
        BLPAPI_CALL_MESSAGE_ADDREF(d_handle);
    }

    return *this;
}

// ACCESSORS
inline Name Message::messageType() const
{
    return Name(blpapi_Message_messageType(d_handle));
}

inline const char *Message::topicName() const
{
    return blpapi_Message_topicName(d_handle);
}

inline Service Message::service() const
{
    return Service(blpapi_Message_service(d_handle));
}

inline int Message::numCorrelationIds() const
{
    return blpapi_Message_numCorrelationIds(d_handle);
}

inline CorrelationId Message::correlationId(size_t index) const
{
    if (index >= (size_t)numCorrelationIds())
        throw IndexOutOfRangeException("index >= numCorrelationIds");
    return CorrelationId(blpapi_Message_correlationId(d_handle, index));
}

inline bool Message::hasElement(
        const char *name, bool excludeNullElements) const
{
    return d_elements.hasElement(Name(name), excludeNullElements);
}

inline bool Message::hasElement(
        const Name& name, bool excludeNullElements) const
{
    return d_elements.hasElement(name, excludeNullElements);
}

inline size_t Message::numElements() const { return d_elements.numElements(); }

inline const Element Message::getElement(const Name& name) const
{
    return d_elements.getElement(name);
}

inline const Element Message::getElement(const char *nameString) const
{
    return d_elements.getElement(Name(nameString));
}

inline bool Message::getElementAsBool(const Name& name) const
{
    return d_elements.getElementAsBool(name);
}

inline bool Message::getElementAsBool(const char *name) const
{
    return d_elements.getElementAsBool(Name(name));
}

inline char Message::getElementAsChar(const Name& name) const
{
    return d_elements.getElementAsChar(name);
}

inline char Message::getElementAsChar(const char *name) const
{
    return d_elements.getElementAsChar(Name(name));
}

inline Int32 Message::getElementAsInt32(const Name& name) const
{
    return d_elements.getElementAsInt32(name);
}

inline Int32 Message::getElementAsInt32(const char *name) const
{
    return d_elements.getElementAsInt32(Name(name));
}

inline Int64 Message::getElementAsInt64(const Name& name) const
{
    return d_elements.getElementAsInt64(name);
}

inline Int64 Message::getElementAsInt64(const char *name) const
{
    return d_elements.getElementAsInt64(Name(name));
}

inline Float32 Message::getElementAsFloat32(const Name& name) const
{
    return d_elements.getElementAsFloat32(name);
}

inline Float32 Message::getElementAsFloat32(const char *name) const
{
    return d_elements.getElementAsFloat32(Name(name));
}

inline Float64 Message::getElementAsFloat64(const Name& name) const
{
    return d_elements.getElementAsFloat64(name);
}

inline Float64 Message::getElementAsFloat64(const char *name) const
{
    return d_elements.getElementAsFloat64(Name(name));
}

inline Datetime Message::getElementAsDatetime(const Name& name) const
{
    return d_elements.getElementAsDatetime(name);
}

inline Datetime Message::getElementAsDatetime(const char *name) const
{
    return d_elements.getElementAsDatetime(Name(name));
}

inline const char *Message::getElementAsString(const Name& name) const
{
    return d_elements.getElementAsString(name);
}

inline const char *Message::getElementAsString(const char *name) const
{
    return d_elements.getElementAsString(Name(name));
}

inline Bytes Message::getElementAsBytes(const Name& name) const
{
    return d_elements.getElementAsBytes(name);
}

inline const char *Message::getRequestId() const
{
    const char *requestId = 0;
    BLPAPI_CALL(blpapi_Message_getRequestId)(d_handle, &requestId);
    return requestId;
}

inline const Element Message::asElement() const { return d_elements; }

inline const char *Message::getPrivateData(size_t *size) const
{
    return blpapi_Message_privateData(d_handle, size);
}

inline Message::Fragment Message::fragmentType() const
{
    return (Message::Fragment)BLPAPI_CALL_MESSAGE_FRAGMENTTYPE(d_handle);
}

inline Message::RecapType::Type Message::recapType() const
{
    return static_cast<Message::RecapType::Type>(
            BLPAPI_CALL(blpapi_Message_recapType)(d_handle));
}

inline int Message::timeReceived(TimePoint *timestamp) const
{
    return BLPAPI_CALL_MESSAGE_TIMERECEIVED(d_handle, timestamp);
}

inline std::ostream& Message::print(
        std::ostream& stream, int level, int spacesPerLevel) const
{
    if (BLPAPI_CALL_AVAILABLE(blpapi_Message_print)) {
        BLPAPI_CALL(blpapi_Message_print)
        (d_handle,
                StreamProxyOstream::writeToStream,
                &stream,
                level,
                spacesPerLevel);
        return stream;
    } else {
        return d_elements.print(stream, level, spacesPerLevel);
    }
}

inline std::ostream& operator<<(std::ostream& stream, const Message& message)
{
    return message.print(stream, 0, -1);
}

inline const blpapi_Message_t *Message::impl() const { return d_handle; }

inline blpapi_Message_t *Message::impl() { return d_handle; }

} // close namespace blpapi
} // close namespace BloombergLP

#endif // #ifdef __cplusplus
#endif // #ifndef INCLUDED_BLPAPI_MESSAGE
