#include <sycl/sycl.hpp>

using namespace sycl::ext::oneapi::experimental;

namespace fpga_tools {

#if __cplusplus >= 202002L

template <auto... Props> using properties_t = decltype(properties{Props...});

#endif

// Type traits to check if a type is annotated_ptr or
// annotated_arg
template <typename T> struct is_annotated_class : std::false_type {};

template <typename T, typename... Props>
struct is_annotated_class<annotated_ptr<T, detail::properties_t<Props...>>>
    : std::true_type {};

template <typename T, typename... Props>
struct is_annotated_class<annotated_arg<T, detail::properties_t<Props...>>>
    : std::true_type {};

// Type traits to get the underlying raw type of annotated_arg/annotated_ptr
template <typename T> struct get_raw_type {};

template <typename T, typename... Props>
struct get_raw_type<annotated_ptr<T, detail::properties_t<Props...>>> {
  using type = T;
};

template <typename T, typename... Props>
struct get_raw_type<annotated_arg<T, detail::properties_t<Props...>>> {
  static constexpr bool is_annotated_arg_for_pointer = false;
  static_assert(is_annotated_arg_for_pointer,
                "'alloc_annotated' cannot be specified with annotated_arg<T> "
                "as template parameter if T is a non-pointer type");
};

template <typename T, typename... Props>
struct get_raw_type<annotated_arg<T *, detail::properties_t<Props...>>> {
  using type = T;
};

// Type traits to get the type of the property list in
// annotated_arg/annotated_ptr
template <typename T> struct get_property_list {};

template <typename T, typename... Props>
struct get_property_list<annotated_ptr<T, detail::properties_t<Props...>>> {
  using type = detail::properties_t<Props...>;
};

template <typename T, typename... Props>
struct get_property_list<annotated_arg<T, detail::properties_t<Props...>>> {
  using type = detail::properties_t<Props...>;
};

// Type traits to remove alignment from a property list. This is needed for
// because the annotated malloc API does not support compile-time alignment
// property
template <typename T> struct remove_align_from {};

template <> struct remove_align_from<detail::empty_properties_t> {
  using type = detail::empty_properties_t;
};

template <typename Prop, typename... Props>
struct remove_align_from<detail::properties_t<Prop, Props...>> {
  using type = std::conditional_t<
      detail::HasAlign<detail::properties_t<Prop>>::value,
      detail::properties_t<Props...>,
      detail::merged_properties_t<
          detail::properties_t<Prop>,
          typename remove_align_from<detail::properties_t<Props...>>::type>>;
};

template <typename T> struct split_annotated_type {
  static constexpr bool is_valid_annotated_type = is_annotated_class<T>::value;
  static_assert(is_valid_annotated_type,
                "alloc_annotated function only takes 'annotated_ptr' or "
                "'annotated_arg' type as a template parameter");

  using raw_type = typename get_raw_type<T>::type;
  using all_properties = typename get_property_list<T>::type;
  static constexpr size_t alignment =
      detail::GetAlignFromPropList<all_properties>::value;
  using properties = typename remove_align_from<all_properties>::type;
};

// Wrapper function that allocates USM host memory with compile-time properties
// and returns annotated_ptr
template <typename T>
T alloc_annotated(size_t count, const sycl::queue &syclQueue,
                  sycl::usm::alloc usmKind = sycl::usm::alloc::host) {
  auto ann_ptr =
      aligned_alloc_annotated<typename split_annotated_type<T>::raw_type,
                              typename split_annotated_type<T>::properties>(
          split_annotated_type<T>::alignment, count, syclQueue, usmKind);

  if (ann_ptr.get() == nullptr) {
    std::cerr << "Memory allocation returns null" << std::endl;
    std::terminate();
  }

  return T{ann_ptr.get()};
}

} // namespace fpga_tools