#ifndef CLCOMMONS_UTIL_TYPE_H
#define CLCOMMONS_UTIL_TYPE_H

#ifdef ENABLE_CL_CPP

template<class T, T v>
struct integral_constant{
  enum{ value = v };
  typedef T value_type;
  typedef integral_constant<T,v> type;
};

typedef integral_constant<bool,false> false_type;
typedef integral_constant<bool,true> true_type;

template<typename T, typename U> struct is_same: false_type{};
template<typename T> struct is_same<T, T> : true_type{};

template<typename T> struct make_unsigned {};
template<> struct make_unsigned<char>{ typedef uchar type; };
template<> struct make_unsigned<short>{ typedef ushort type; };
template<> struct make_unsigned<int>{ typedef uint type; };
template<> struct make_unsigned<long>{ typedef ulong type; };
template<> struct make_unsigned<uchar>{ typedef uchar type; };
template<> struct make_unsigned<ushort>{ typedef ushort type; };
template<> struct make_unsigned<uint>{ typedef uint type; };
template<> struct make_unsigned<ulong>{ typedef ulong type; };

template<typename T> struct make_signed {};
template<> struct make_signed<char>{ typedef char type; };
template<> struct make_signed<short>{ typedef short type; };
template<> struct make_signed<int>{ typedef int type; };
template<> struct make_signed<long>{ typedef long type; };
template<> struct make_signed<uchar>{ typedef uchar type; };
template<> struct make_signed<ushort>{ typedef ushort type; };
template<> struct make_signed<uint>{ typedef uint type; };
template<> struct make_signed<ulong>{ typedef ulong type; };


template<bool IF, typename ThenType, typename ElseType>
struct static_if{
    typedef ThenType Type;      // true
};

template<typename ThenType, typename ElseType>
struct static_if<false, ThenType, ElseType>{
    typedef ElseType Type;      // false
};


template<typename A, typename B>
struct is_same{
    enum {
        VALUE = 0,
        NEGATE = 1
    };
};

template<typename A>
struct is_same <A, A>{
    enum {
        VALUE = 1,
        NEGATE = 0
    };
};

struct NullType{
    template<typename T>
    inline NullType& operator =(const T& b) { return *this; }
};


template<int A>
struct Int2Type{
   enum {VALUE = A};
};


template<int N, int CURRENT_VAL = N, int COUNT = 0>
struct Log2{
    /// Static logarithm value
    enum { VALUE = Log2<N, (CURRENT_VAL >> 1), COUNT + 1>::VALUE };         // Inductive case
};

template<int N, int COUNT>
struct Log2<N, 0, COUNT>{
    enum {VALUE = (1 << (COUNT - 1) < N) ?                                  // Base case
        COUNT :
        COUNT - 1 };
};


template<int N>
struct ISPowerOfTwo{
    enum { VALUE = ((N & (N - 1)) == 0) };
};


template<typename Tp>
struct is_pointer{
    enum { VALUE = 0 };
};

struct is_pointer<Tp*>{
    enum { VALUE = 1 };
};


template<typename Tp>
struct is_volatile{
    enum { VALUE = 0 };
};

template<typename Tp>
struct is_volatile<Tp volatile>{
    enum { VALUE = 1 };
};


template<typename Tp, typename Up = Tp>
struct remove_cv{
    typedef Up Type;
};

template<typename Tp, typename Up>
struct remove_cv<Tp, volatile Up>{
    typedef Up Type;
};

template<typename Tp, typename Up>
struct remove_cv<Tp, const Up>
{
    typedef Up Type;
};

template<typename Tp, typename Up>
struct remove_cv<Tp, const volatile Up>
{
    typedef Up Type;
};


template<bool Condition, class T = void>
struct enable_if{
    /// Enable-if type for SFINAE dummy variables
    typedef T Type;
};

template<class T>
struct enable_if<false, T> {};

enum NumericCategory{
    NOT_A_NUMBER,
    SIGNED_INTEGER,
    UNSIGNED_INTEGER,
    FLOATING_POINT
};


template<NumericCategory _CATEGORY, bool _PRIMITIVE, bool _NULL_TYPE, typename _UnsignedBits>
struct BaseTraits{
    /// NumericCategory
    enum{
        CATEGORY        = _CATEGORY,
        PRIMITIVE       = _PRIMITIVE,
        NULL_TYPE       = _NULL_TYPE,
    };
};

template<typename _UnsignedBits>
struct BaseTraits<UNSIGNED_INTEGER, true, false, _UnsignedBits>
{
    typedef _UnsignedBits       UnsignedBits;

    //static const UnsignedBits   MIN_VALUE     = UnsignedBits(0);
    //static const UnsignedBits   MAX_VALUE     = UnsignedBits(-1);

    static const UnsignedBits   MIN_VALUE(){ return UnsignedBits(0); }
    static const UnsignedBits   MAX_VALUE(){ return UnsignedBits(-1); }

    enum
    {
        CATEGORY        = UNSIGNED_INTEGER,
        PRIMITIVE       = true,
        NULL_TYPE       = false,
    };


    static inline UnsignedBits TwiddleIn(UnsignedBits key)
    {
        return key;
    }

    static inline UnsignedBits TwiddleOut(UnsignedBits key)
    {
        return key;
    }
};


template<typename _UnsignedBits>
struct BaseTraits<SIGNED_INTEGER, true, false, _UnsignedBits>
{
    typedef _UnsignedBits       UnsignedBits;

    static const UnsignedBits   HIGH_BIT(){ return UnsignedBits(1) << ((sizeof(UnsignedBits) * 8) - 1); }
    static const UnsignedBits   MIN_VALUE(){ return HIGH_BIT(); }
    static const UnsignedBits   MAX_VALUE(){ return UnsignedBits(-1) ^ HIGH_BIT(); }

    enum
    {
        CATEGORY        = SIGNED_INTEGER,
        PRIMITIVE       = true,
        NULL_TYPE       = false,
    };

    static inline UnsignedBits TwiddleIn(UnsignedBits key)
    {
        //on negative: flip sign (twos complement) and complement with max value to preserve ordering, on positive: add MSB
        return (key & HIGH_BIT()) ? (HIGH_BIT() - 1) - ((~key) + 1) : (HIGH_BIT() | key);
    };

    static inline UnsignedBits TwiddleOut(UnsignedBits key)
    {
        return (key & HIGH_BIT()) ? (HIGH_BIT() - 1) - ((~key) + 1): ((HIGH_BIT() - 1) & key);
    };

};

template<typename _UnsignedBits>
struct BaseTraits<FLOATING_POINT, true, false, _UnsignedBits>
{
    typedef _UnsignedBits       UnsignedBits;
    typedef make_signed<UnsignedBits>       SignedBits;

    static const UnsignedBits   HIGH_BIT(){ return UnsignedBits(1) << ((sizeof(UnsignedBits) * 8) - 1); }
    static const UnsignedBits   MIN_VALUE(){ return UnsignedBits(-1); }
    static const UnsignedBits   MAX_VALUE(){ return UnsignedBits(-1) ^ HIGH_BIT; }


    static inline UnsignedBits TwiddleIn(UnsignedBits key)
    {
        UnsignedBits mask = (key & HIGH_BIT()) ? UnsignedBits(-1) : HIGH_BIT();
        return key ^ mask;
    };

    static inline UnsignedBits TwiddleOut(UnsignedBits key)
    {
        UnsignedBits mask = (key & HIGH_BIT()) ? HIGH_BIT() : UnsignedBits(-1);
        return key ^ mask;
    };

    enum
    {
        CATEGORY        = FLOATING_POINT,
        PRIMITIVE       = true,
        NULL_TYPE       = false,
    };
};

template<typename T> struct numeric_traits :            BaseTraits<NOT_A_NUMBER, false, false, T> {};

template<> struct numeric_traits<NullType> :            BaseTraits<NOT_A_NUMBER, false, true, NullType> {};

//template<> struct numeric_traits<char> :                BaseTraits<SIGNED_INTEGER, true, false, unsigned char> {};
template<> struct numeric_traits<signed char> :         BaseTraits<SIGNED_INTEGER, true, false, uchar> {};
template<> struct numeric_traits<short> :               BaseTraits<SIGNED_INTEGER, true, false, ushort> {};
template<> struct numeric_traits<int> :                 BaseTraits<SIGNED_INTEGER, true, false, uint> {};
template<> struct numeric_traits<long> :                BaseTraits<SIGNED_INTEGER, true, false, ulong> {};

template<> struct numeric_traits<uchar> :       BaseTraits<UNSIGNED_INTEGER, true, false, uchar> {};
template<> struct numeric_traits<ushort> :      BaseTraits<UNSIGNED_INTEGER, true, false, ushort> {};
template<> struct numeric_traits<uint> :        BaseTraits<UNSIGNED_INTEGER, true, false, uint> {};
template<> struct numeric_traits<ulong> :       BaseTraits<UNSIGNED_INTEGER, true, false, ulong> {};

template<> struct numeric_traits<float> :               BaseTraits<FLOATING_POINT, true, false, uint> {};
template<> struct numeric_traits<double> :              BaseTraits<FLOATING_POINT, true, false, ulong> {};


template<typename T> struct underlying_scalar_type {};
template<> struct underlying_scalar_type<char>{ typedef char type; };
template<> struct underlying_scalar_type<char2>{ typedef char type; };
template<> struct underlying_scalar_type<char4>{ typedef char type; };
template<> struct underlying_scalar_type<char8>{ typedef char type; };
template<> struct underlying_scalar_type<char16>{ typedef char type; };
template<> struct underlying_scalar_type<uchar>{ typedef uchar type; };
template<> struct underlying_scalar_type<uchar2>{ typedef uchar type; };
template<> struct underlying_scalar_type<uchar4>{ typedef uchar type; };
template<> struct underlying_scalar_type<uchar8>{ typedef uchar type; };
template<> struct underlying_scalar_type<uchar16>{ typedef uchar type; };

template<> struct underlying_scalar_type<short>{ typedef short type; };
template<> struct underlying_scalar_type<short2>{ typedef short type; };
template<> struct underlying_scalar_type<short4>{ typedef short type; };
template<> struct underlying_scalar_type<short8>{ typedef short type; };
template<> struct underlying_scalar_type<short16>{ typedef short type; };
template<> struct underlying_scalar_type<ushort>{ typedef ushort type; };
template<> struct underlying_scalar_type<ushort2>{ typedef ushort type; };
template<> struct underlying_scalar_type<ushort4>{ typedef ushort type; };
template<> struct underlying_scalar_type<ushort8>{ typedef ushort type; };
template<> struct underlying_scalar_type<ushort16>{ typedef ushort type; };

template<> struct underlying_scalar_type<int>{ typedef int type; };
template<> struct underlying_scalar_type<int2>{ typedef int type; };
template<> struct underlying_scalar_type<int4>{ typedef int type; };
template<> struct underlying_scalar_type<int8>{ typedef int type; };
template<> struct underlying_scalar_type<int16>{ typedef int type; };
template<> struct underlying_scalar_type<uint>{ typedef uint type; };
template<> struct underlying_scalar_type<uint2>{ typedef uint type; };
template<> struct underlying_scalar_type<uint4>{ typedef uint type; };
template<> struct underlying_scalar_type<uint8>{ typedef uint type; };
template<> struct underlying_scalar_type<uint16>{ typedef uint type; };

template<> struct underlying_scalar_type<long>{ typedef long type; };
template<> struct underlying_scalar_type<long2>{ typedef long type; };
template<> struct underlying_scalar_type<long4>{ typedef long type; };
template<> struct underlying_scalar_type<long8>{ typedef long type; };
template<> struct underlying_scalar_type<long16>{ typedef long type; };
template<> struct underlying_scalar_type<ulong>{ typedef ulong type; };
template<> struct underlying_scalar_type<ulong2>{ typedef ulong type; };
template<> struct underlying_scalar_type<ulong4>{ typedef ulong type; };
template<> struct underlying_scalar_type<ulong8>{ typedef ulong type; };
template<> struct underlying_scalar_type<ulong16>{ typedef ulong type; };

template<> struct underlying_scalar_type<float>{ typedef float type; };
template<> struct underlying_scalar_type<float2>{ typedef float type; };
template<> struct underlying_scalar_type<float4>{ typedef float type; };
template<> struct underlying_scalar_type<float8>{ typedef float type; };
template<> struct underlying_scalar_type<float16>{ typedef float type; };

template<> struct underlying_scalar_type<double>{ typedef double type; };
template<> struct underlying_scalar_type<double2>{ typedef double type; };
template<> struct underlying_scalar_type<double4>{ typedef double type; };
template<> struct underlying_scalar_type<double8>{ typedef double type; };
template<> struct underlying_scalar_type<double16>{ typedef double type; };

//template<typename T> struct is_vec2 : false_type{};
//template<> struct is_vec2<char2> : true_type{};
//template<> struct is_vec2<short2> : true_type{};
//template<> struct is_vec2<int2> : true_type{};
//template<> struct is_vec2<long2> : true_type{};
//template<> struct is_vec2<uchar2> : true_type{};
//template<> struct is_vec2<ushort2> : true_type{};
//template<> struct is_vec2<uint2> : true_type{};
//template<> struct is_vec2<ulong2> : true_type{};
//template<> struct is_vec2<float2> : true_type{};
//template<> struct is_vec2<double2> : true_type{};

template<typename T>
struct fast_lds_type{
    typedef T value_type;
};

template<>
struct fast_lds_type<uchar>{ typedef uint value_type; };
template<>
struct fast_lds_type<ushort>{ typedef uint value_type; };
template<>
struct fast_lds_type<char>{ typedef int value_type; };
template<>
struct fast_lds_type<short>{ typedef int value_type; };

#endif

#endif
